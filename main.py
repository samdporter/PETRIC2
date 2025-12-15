"""Main file to modify for submissions.

Once renamed or symlinked as `main.py`, it will be used by `petric.py` as follows:

>>> from main import Submission, submission_callbacks
>>> from petric import data, metrics
>>> algorithm = Submission(data)
>>> algorithm.run(np.inf, callbacks=metrics + submission_callbacks)
"""
from __future__ import annotations

from typing import Iterable, List, Sequence

from cil.optimisation.algorithms import Algorithm
from cil.optimisation.functions import IndicatorBox
from cil.optimisation.utilities import Preconditioner, Sampler, callbacks
from petric import Dataset
from sirf.contrib.partitioner import partitioner


class MaxIteration(callbacks.Callback):
    """
    The organisers try to `Submission(data).run(inf)` i.e. for infinite iterations (until timeout).
    This callback forces stopping after `max_iteration` instead.
    """

    def __init__(self, max_iteration: int, verbose: int = 1):
        super().__init__(verbose)
        self.max_iteration = max_iteration

    def __call__(self, algorithm: Algorithm):
        if algorithm.iteration >= self.max_iteration:
            raise StopIteration


class MyPreconditioner(Preconditioner):
    """Lehmer-mean preconditioner combining prior, EM, and optional data Hessian diagonals."""

    def __init__(
        self,
        kappa,
        prior=None,
        obj_funs: Sequence | None = None,
        ref_image=None,
        p: float = 2.0,
        mix_data: bool = False,
        epsilon: float = 1e-6,
    ):
        self.p = p
        self.epsilon = epsilon
        self.d_prior = self._compute_prior_diag(kappa, prior, ref_image, epsilon)
        self.d_em = self._compute_em_diag(obj_funs, epsilon)
        if self.d_em is None:
            self.d_em = self.d_prior.clone() if hasattr(self.d_prior, "clone") else self.d_prior
        self.d_data = self._compute_data_diag(obj_funs, ref_image, epsilon) if mix_data else None

        base_mean = self._lehmer_mean(self.d_prior, self.d_em, p=p, eps=epsilon)
        if mix_data and self.d_data is not None:
            base_mean = 0.5 * base_mean + 0.5 * self.d_data
        self.diag = base_mean + epsilon

    @staticmethod
    def _lehmer_mean(x, y, p: float = 2.0, eps: float = 1e-6):
        numerator = (x ** (p - 1)) + (y ** (p - 1))
        denominator = (x ** p) + (y ** p) + eps
        return numerator / denominator

    def _compute_prior_diag(self, kappa, prior, ref_image, eps):
        if prior is not None and ref_image is not None:
            try:
                ones = ref_image.get_uniform_copy(1.0)
                return prior.multiply_with_Hessian(ref_image, ones)
            except Exception:
                pass
        # fallback to kappa^2
        return kappa * kappa + eps

    def _compute_em_diag(self, obj_funs: Sequence | None, eps):
        if not obj_funs:
            return None
        sensitivity = None
        for f in obj_funs:
            try:
                sens = f.get_subset_sensitivity(0)
            except Exception:
                continue
            sensitivity = sens.clone() if sensitivity is None else sensitivity + sens
        if sensitivity is None:
            return None
        return sensitivity + eps

    def _compute_data_diag(self, obj_funs: Sequence | None, ref_image, eps):
        if not obj_funs or ref_image is None:
            return None
        diag = None
        ones = ref_image.get_uniform_copy(1.0)
        for f in obj_funs:
            try:
                hess_vec = f.multiply_with_Hessian(ref_image, ones)
            except Exception:
                continue
            diag = hess_vec.clone() if diag is None else diag + hess_vec
        if diag is None:
            return None
        diag *= 1.0 / len(obj_funs)
        return diag + eps

    def apply(self, algorithm, gradient, out=None):
        return gradient.divide(self.diag, out=out)


class LazyStochasticLBFGSB(Algorithm):
    """Lazy stochastic L-BFGS-B supporting SAGA and SVRG style gradients.

    The implementation mirrors the structure of CIL proximal algorithms: a
    constraint function ``g`` (defaulting to ``IndicatorBox`` for non-negativity)
    provides the projection via its proximal operator, while the stochastic
    objective components are supplied as a sequence of differentiable
    ``Function`` instances. This lets us re-use CIL's function interface instead
    of reinventing gradient/`proximal` plumbing.
    """

    def __init__(
        self,
        initial,
        obj_funs: Sequence,
        mode: str = "SAGA",
        step_size: float = 0.1,
        decay: float = 0.0,
        memory: int = 5,
        lazy_interval: int | None = None,
        preconditioner: Preconditioner | None = None,
        sampler: Sampler | None = None,
        update_objective_interval: int = 10,
        g: IndicatorBox | None = None,
    ):
        super().__init__()
        self.x = initial.clone()
        self.obj_funs = list(obj_funs)
        self.mode = mode.upper()
        self.step_size0 = step_size
        self.decay = decay
        self.memory = memory
        self.lazy_interval = lazy_interval or len(self.obj_funs)
        self.preconditioner = preconditioner
        self.sampler = sampler or Sampler.random_without_replacement(len(self.obj_funs))
        self.update_objective_interval = update_objective_interval
        self.max_iteration = 0
        self.g = g if g is not None else IndicatorBox(lower=0, accelerated=False)

        # L-BFGS memory
        self.s_list: List = []
        self.y_list: List = []

        # stochastic state
        if self.mode == "SAGA":
            self._init_saga_tables()
        else:
            self._init_svrg_snapshot()
        self.anchor_grad = self._current_global_grad().clone()
        self.loss = []

    # ---- initialisation helpers ----
    def _init_saga_tables(self):
        self.stored_grads = []
        for f in self.obj_funs:
            g = f.gradient(self.x)
            self.stored_grads.append(g.clone())
        self.mean_grad = self._mean_gradients(self.stored_grads)

    def _init_svrg_snapshot(self):
        self.snapshot_x = self.x.clone()
        self.snapshot_grads = []
        for f in self.obj_funs:
            g = f.gradient(self.snapshot_x)
            self.snapshot_grads.append(g.clone())
        self.mu = self._mean_gradients(self.snapshot_grads)

    def _mean_gradients(self, grads: Iterable):
        total = None
        for g in grads:
            total = g.clone() if total is None else total + g
        return total * (1.0 / len(self.obj_funs))

    def _current_global_grad(self):
        grads = [f.gradient(self.x) for f in self.obj_funs]
        return self._mean_gradients(grads)

    # ---- core algorithm ----
    def step_size(self):
        if self.decay <= 0:
            return self.step_size0
        return self.step_size0 / (1.0 + self.decay * self.iteration)

    def _sample_index(self):
        if hasattr(self.sampler, "next"):
            return self.sampler.next()
        return next(self.sampler)

    def _stochastic_gradient(self, idx):
        if self.mode == "SAGA":
            grad_i = self.obj_funs[idx].gradient(self.x)
            estimator = grad_i - self.stored_grads[idx] + self.mean_grad
            # update table
            self.mean_grad += (grad_i - self.stored_grads[idx]) * (1.0 / len(self.obj_funs))
            self.stored_grads[idx] = grad_i.clone()
            return estimator
        # SVRG
        grad_i = self.obj_funs[idx].gradient(self.x)
        estimator = grad_i - self.snapshot_grads[idx] + self.mu
        return estimator

    def _two_loop(self, grad):
        q = grad.clone()
        alphas = []
        rhos = []
        for s, y in zip(reversed(self.s_list), reversed(self.y_list)):
            rho = 1.0 / (y.dot(s) + 1e-12)
            alpha = rho * s.dot(q)
            q -= alpha * y
            alphas.append(alpha)
            rhos.append(rho)
        # initial Hessian application via preconditioner
        r = q.clone()
        if self.preconditioner is not None:
            r = self.preconditioner.apply(self, q)
        # forward loop
        for (s, y), alpha, rho in zip(zip(self.s_list, self.y_list), reversed(alphas), reversed(rhos)):
            beta = rho * y.dot(r)
            r += s * (alpha - beta)
        return -r

    def _project_nonneg(self, x):
        # reuse CIL proximal operator when available to stay within the standard
        # Algorithm + Function contract
        if hasattr(self.g, "proximal"):
            return self.g.proximal(1.0, x)
        return x.maximum(0, out=x.clone())

    def _maybe_update_memory(self, prev_x, prev_anchor_grad):
        if (self.iteration + 1) % self.lazy_interval != 0:
            return
        if self.mode == "SAGA":
            current_grad = self.mean_grad.clone()
        else:
            current_grad = self.mu.clone()
            # refresh snapshot for next epoch
            self.snapshot_x = self.x.clone()
            self.snapshot_grads = []
            for f in self.obj_funs:
                g = f.gradient(self.snapshot_x)
                self.snapshot_grads.append(g.clone())
            self.mu = self._mean_gradients(self.snapshot_grads)
        y = current_grad - prev_anchor_grad
        s = self.x - prev_x
        if y.dot(s) <= 0:
            return
        self.s_list.append(s)
        self.y_list.append(y)
        if len(self.s_list) > self.memory:
            self.s_list.pop(0)
            self.y_list.pop(0)
        # update anchor
        self.anchor_grad = current_grad

    def update(self):
        idx = self._sample_index()
        grad_est = self._stochastic_gradient(idx)
        direction = self._two_loop(grad_est)
        step = self.step_size()
        prev_x = self.x.clone()
        prev_anchor = self.anchor_grad.clone()
        self.x = self._project_nonneg(self.x + step * direction)
        self._maybe_update_memory(prev_x, prev_anchor)
        if self.iteration % self.update_objective_interval == 0:
            self.loss.append(self.objective())

    def objective(self):
        total = 0.0
        for f in self.obj_funs:
            total += f(self.x)
        return total

    def get_last_loss(self):
        return self.loss[-1] if self.loss else None

    def run(self, max_iteration: int, callbacks: Iterable = ()):  # pragma: no cover - loop control
        self.max_iteration = int(max_iteration)
        self.iteration = 0
        cb_list = callbacks or []
        try:
            while self.iteration < self.max_iteration:
                for cb in cb_list:
                    cb(self)
                self.update()
                self.iteration += 1
        except StopIteration:
            pass
        return self.x


class Submission(LazyStochasticLBFGSB):
    """Lazy stochastic quasi-Newton solver with non-negativity constraints."""

    def __init__(self, data: Dataset, num_subsets: int = 7, step_size: float = 0.1, update_objective_interval: int = 10,
                 mode: str = "SAGA", decay: float = 0.0, lazy_interval: int | None = None, memory: int = 5,
                 mix_data_preconditioner: bool = True):
        data_sub, acq_models, obj_funs = partitioner.data_partition(
            data.acquired_data, data.additive_term, data.mult_factors, num_subsets, mode='staggered',
            initial_image=data.OSEM_image)
        data.prior.set_penalisation_factor(data.prior.get_penalisation_factor() / len(obj_funs))
        data.prior.set_up(data.OSEM_image)
        for f in obj_funs:
            f.set_prior(data.prior)

        obj_funs_for_precond = list(obj_funs)

        # Use the same sign convention as the reference ISTA implementation (maximisation rewritten as minimisation)
        obj_funs = [(-1) * f for f in obj_funs]
        sampler = Sampler.random_without_replacement(len(obj_funs))
        preconditioner = MyPreconditioner(
            kappa=data.kappa,
            prior=data.prior,
            obj_funs=obj_funs_for_precond,
            ref_image=data.OSEM_image,
            mix_data=mix_data_preconditioner,
        )
        super().__init__(
            initial=data.OSEM_image,
            obj_funs=obj_funs,
            mode=mode,
            step_size=step_size,
            decay=decay,
            memory=memory,
            lazy_interval=lazy_interval,
            preconditioner=preconditioner,
            sampler=sampler,
            update_objective_interval=update_objective_interval,
            g=IndicatorBox(lower=0, accelerated=False),
        )

    def objective(self):
        # use IndicatorBox to enforce constraint contribution if accelerated flag handles
        base = super().objective()
        if hasattr(self.g, "__call__"):
            return base + self.g(self.x)
        return base


submission_callbacks = [MaxIteration(1000)]
