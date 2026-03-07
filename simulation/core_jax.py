"""JAX primitives for continuous-Bernoulli dynamics on [-1, 1]."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import jit


@jit
def log_partition(theta: jnp.ndarray) -> jnp.ndarray:
    """Log-partition psi(theta) = log(2*sinh(theta)/theta) with stable branches."""
    a = jnp.abs(theta)
    t2 = theta**2
    # psi(theta) = log(2) + theta^2/6 - theta^4/180 + O(theta^6)
    taylor = jnp.log(2.0) + t2 / 6.0 - (t2**2) / 180.0
    safe = jnp.where(a < 1e-6, jnp.ones_like(theta), theta)
    mid = jnp.log(2.0 * jnp.sinh(safe) / safe)
    large = a - jnp.log(a + 1e-30)
    return jnp.where(a < 0.1, taylor, jnp.where(a < 20.0, mid, large))


@jit
def langevin(theta: jnp.ndarray) -> jnp.ndarray:
    """Langevin function L(theta) = coth(theta) - 1/theta with stable near-zero branch."""
    a = jnp.abs(theta)
    t2 = theta**2
    taylor = theta / 3.0 - theta * t2 / 45.0 + 2.0 * theta * (t2**2) / 945.0
    safe = jnp.where(a < 1e-6, jnp.ones_like(theta), theta)
    exact = 1.0 / jnp.tanh(safe) - 1.0 / safe
    return jnp.where(a < 0.1, taylor, exact)


@jit
def fisher_metric(theta: jnp.ndarray) -> jnp.ndarray:
    """Fisher diagonal g(theta) = 1/theta^2 - csch(theta)^2."""
    a = jnp.abs(theta)
    t2 = theta**2
    taylor = (
        1.0 / 3.0
        - t2 / 15.0
        + 2.0 * (t2**2) / 189.0
        - (t2**3) / 675.0
        + 2.0 * (t2**4) / 10395.0
    )
    safe = jnp.where(a < 0.5, jnp.ones_like(theta), theta)
    e = jnp.exp(-2.0 * jnp.abs(safe))
    inv_sinh2 = 4.0 * e / (1.0 - e) ** 2
    direct = 1.0 / (safe**2) - inv_sinh2
    return jnp.where(a < 0.5, taylor, direct)


@jit
def sample_cb(theta: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
    """Sample from continuous Bernoulli on [-1, 1] via inverse CDF."""
    u = jax.random.uniform(key, theta.shape, minval=1e-7, maxval=1.0 - 1e-7)
    a = jnp.abs(theta)
    safe = jnp.where(a < 1e-6, jnp.ones_like(theta), theta)
    sample = jnp.where(
        a < 1e-6,
        2.0 * u - 1.0,
        jnp.logaddexp(safe + jnp.log(u), -safe + jnp.log1p(-u)) / safe,
    )
    return jnp.clip(sample, -1.0 + 1e-7, 1.0 - 1e-7)


@jit
def total_vfe(theta: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Simple per-step free-energy proxy used for monitoring."""
    return jnp.sum(log_partition(theta) - theta * x)

