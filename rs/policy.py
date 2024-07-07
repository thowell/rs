import jax
import jax.numpy as jnp
from mujoco.mjx._src import dataclasses
from typing import Tuple


class Policy(dataclasses.PyTreeNode):
  """Affine feedback policy.

  output = clip(weight @ ((input - shift) / scale), limits[0], limits[1])

  Attributes:
    weight: feedback matrix
    shift: input shift
    scale: input scaling
    limits: output limits
  """

  weight: jax.Array
  shift: jax.Array
  scale: jax.Array
  limits: Tuple[jax.Array, jax.Array]


def initialize_policy(
    nact: int, nobs: int, limits: Tuple[jax.Array, jax.Array]
) -> Policy:
  """Initialize policy.

  Args:
      nact (int): action dimension
      nobs (int): observation dimension
      limits (Tuple[jax.Array, jax.Array]): action limits

  Returns:
      Policy
  """
  return Policy(
      weight=jnp.zeros((nact, nobs)),
      shift=jnp.zeros(nobs),
      scale=jnp.ones(nobs),
      limits=limits,
  )


def policy(p: Policy, obs: jax.Array) -> jax.Array:
  """Evaluate policy.

  Args:
      p (Policy)
      obs (jax.Array): input to policy

  Returns:
      jax.Array: output from policy
  """
  return jnp.clip(
      p.weight @ ((obs - p.shift) / (p.scale + 1.0e-5)),
      a_min=p.limits[0],
      a_max=p.limits[1],
  )


def noisy_policy(p: Policy, scale: float, rng) -> Tuple[Policy, jax.Array]:
  """Sample noisy policy.

  Args:
      p (Policy)
      scale (float): scaling
      rng (jax.Array): JAX random number key

  Returns:
      Policy
      perturb (jax.Array): perturbation
  """
  # sample noise: perturb ~ N(0, I)
  perturb = jax.random.normal(rng, shape=p.weight.shape)

  # copy policy
  noisy_policy = Policy(
      weight=p.weight.copy() + scale * perturb,
      shift=p.shift.copy(),
      scale=p.scale.copy(),
      limits=(p.limits[0], p.limits[1]),
  )
  return noisy_policy, perturb


# jit and vmap noisy_policy
v_noisy_policy = jax.jit(jax.vmap(noisy_policy, in_axes=(None, 0, 0)))
