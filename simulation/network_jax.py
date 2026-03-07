"""Efficient JAX implementation matching the manuscript update rules."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit

from simulation.core_jax import fisher_metric, langevin, sample_cb, total_vfe


@partial(jit, static_argnums=(6, 7, 8, 9))
def _train(
    W: jnp.ndarray,
    b: jnp.ndarray,
    data: jnp.ndarray,
    beta: float,
    lr: float,
    key: jax.Array,
    epochs: int,
    steps: int,
    precision_weighted: bool,
    zero_diagonal: bool,
):
    """Train with the same local stochastic rule as the reference implementation."""
    n_patterns = data.shape[0]
    n = W.shape[0]
    mask = (1.0 - jnp.eye(n, dtype=W.dtype)) if zero_diagonal else jnp.ones_like(W)

    def epoch_step(carry, _):
        W, key = carry
        key, k_idx, k_loop = jax.random.split(key, 3)
        idx = jax.random.randint(k_idx, shape=(), minval=0, maxval=n_patterns)
        u = data[idx]

        def inner_step(inner_carry, _):
            W, x_prev, key = inner_carry
            h = W @ x_prev
            theta = beta * (b + h + u)
            key, k_sample = jax.random.split(key)
            x_new = sample_cb(theta, k_sample)
            eps = x_new - langevin(h)

            if precision_weighted:
                g = fisher_metric(beta * h)
                pi = 1.0 / jnp.clip(g, 1e-8)
            else:
                pi = jnp.ones_like(x_new)

            dW = lr * jnp.outer(pi * eps, x_new) * mask
            vfe = total_vfe(theta, x_new)
            return (W + dW, x_new, key), vfe

        init_x = jnp.zeros((n,), dtype=W.dtype)
        (W, _, key), vfes = jax.lax.scan(inner_step, (W, init_x, k_loop), None, length=steps)
        return (W, key), vfes

    (W, key), vfe_hist = jax.lax.scan(epoch_step, (W, key), None, length=epochs)
    return W, vfe_hist.reshape(-1), key


@partial(jit, static_argnums=(6, 7))
def _infer(
    W: jnp.ndarray,
    b: jnp.ndarray,
    x0: jnp.ndarray,
    u: jnp.ndarray,
    beta: float,
    key: jax.Array,
    steps: int,
    stochastic: bool,
):
    def one_step(carry, _):
        x_prev, key = carry
        theta = beta * (b + W @ x_prev + u)
        key, k_sample = jax.random.split(key)
        x_new = jax.lax.cond(stochastic, lambda _: sample_cb(theta, k_sample), lambda _: langevin(theta), operand=None)
        vfe = total_vfe(theta, x_new)
        return (x_new, key), (x_new, vfe)

    (_, key), (acts, vfes) = jax.lax.scan(one_step, (x0, key), None, length=steps)
    return acts, vfes, key


@dataclass
class JAXAttractorNetwork:
    """JAX attractor network with manuscript-equivalent update dynamics."""

    n_nodes: int
    seed: int = 0

    def __post_init__(self):
        self.W = jnp.zeros((self.n_nodes, self.n_nodes), dtype=jnp.float32)
        self.b = jnp.zeros((self.n_nodes,), dtype=jnp.float32)
        self._key = jax.random.PRNGKey(self.seed)

    def train(
        self,
        data,
        *,
        evidence_level: float = 1.0,
        beta: float = 0.1,
        lr: float = 0.01,
        epochs: int = 100,
        steps: int = 10,
        precision_weighted: bool = False,
        zero_diagonal: bool = True,
    ):
        data = jnp.asarray(data, dtype=self.W.dtype) * evidence_level
        self._key, sk = jax.random.split(self._key)
        self.W, vfe_hist, self._key = _train(
            self.W,
            self.b,
            data,
            float(beta),
            float(lr),
            sk,
            int(epochs),
            int(steps),
            bool(precision_weighted),
            bool(zero_diagonal),
        )
        return vfe_hist

    def infer(
        self,
        *,
        x0=None,
        u=None,
        beta: float = 1.0,
        steps: int = 100,
        stochastic: bool = True,
    ):
        x0 = jnp.zeros((self.n_nodes,), dtype=self.W.dtype) if x0 is None else jnp.asarray(x0, dtype=self.W.dtype)
        u = jnp.zeros((self.n_nodes,), dtype=self.W.dtype) if u is None else jnp.asarray(u, dtype=self.W.dtype)
        self._key, sk = jax.random.split(self._key)
        acts, vfes, self._key = _infer(self.W, self.b, x0, u, float(beta), sk, int(steps), bool(stochastic))
        return acts, vfes

    @property
    def S(self):
        return (self.W + self.W.T) / 2.0

    @property
    def A(self):
        return (self.W - self.W.T) / 2.0

