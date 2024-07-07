from functools import partial
import jax
import jax.numpy as jnp
from mujoco import mjx
from mujoco.mjx._src import dataclasses, forward

from rs.policy import Policy, policy
from typing import Callable, Tuple


class Environment(dataclasses.PyTreeNode):
  """Learning environment.

  Attributes:
    model: MuJoCo model
    reward: function returning per-timestep reward
    observation: function returning per-timestep observation
    reset: function reseting environment, returns mjx.Data
    done: function determining environment termination
    naction: number of actions
    nobservation: number of observations
    ndecimation: number of physics steps per policy step
  """

  model: mjx.Model
  data: mjx.Data
  reward: Callable
  observation: Callable
  reset: Callable
  done: Callable
  naction: int
  nobservation: int
  ndecimation: int


@partial(jax.jit, static_argnums=(3,))
def multistep(
    model: mjx.Model, data: mjx.Data, action: jax.Array, nstep: int
) -> mjx.Data:
  """Multiple physics steps for action.

  Args:
      model (mjx.Model)
      data (mjx.Data)
      action (jax.Array)
      nstep (int): number of physics steps.

  Returns:
      mjx.Data
  """
  # set action
  data = data.replace(ctrl=action)

  # step physics
  def step(d, _):
    # step dynamics
    d = forward.step(model, d)

    return d, None

  # next data
  data, _ = jax.lax.scan(step, data, None, length=nstep, unroll=nstep)
  return data

def rollout(
    env: Environment,
    p: Policy,
    d: mjx.Data,
    shift: float = 0.0,
    nhorizon: int = 1000,
) -> Tuple[float, jax.Array]:
  """Simulate environment.

  Args:
      env (Environment): simulation environment
      p (Policy): affine feedback policy
      d (mjx.Data): MuJoCo data
      shift (float): subtract value from per-timestep reward
      nhorizon: number of environment steps
      rng: JAX random number key

  Returns:
      Tuple[jax.Array, jax.Array]: per-timestep rewards and observations
  """
  # get observation
  obs = env.observation(env.model, d)

  # initialize observation statistics
  # (mean, var, count)
  obs_stats_init = (obs.copy(), jnp.zeros_like(obs), 1)

  # continue physics steps
  def continue_step(carry):
    # unpack carry
    data, total_reward, obs, obs_stats, steps, done = carry
    return jnp.logical_and(jnp.where(done, False, True), steps < nhorizon)
  
  # step
  def step(carry):
    # unpack carry
    data, total_reward, obs, obs_stats, steps, done = carry

    # get action
    action = policy(p, obs)

    # step
    next_data = multistep(env.model, data, action, env.ndecimation)

    # done
    next_done = jnp.logical_or(env.done(env.model, next_data), done)
    not_done = jnp.where(next_done, 0, 1)

    # get reward
    reward = env.reward(
        env.model, data, next_data, env.model.opt.timestep * env.ndecimation
    )
    reward -= shift
    reward *= not_done

    # get observation
    next_obs = env.observation(env.model, next_data)
    
    # unpack observation statistics
    mean, var, count = obs_stats

    # update observation statistics
    delta = next_obs - mean
    next_count = count + not_done
    next_mean = mean + not_done * delta / next_count
    next_var = var + not_done * delta * delta * count / next_count
    next_obs_stats = (next_mean, next_var, next_count)

    return (
        next_data,
        total_reward + reward,
        next_obs,
        next_obs_stats,
        steps + 1,
        next_done,
    )

  # loop
  carry = jax.lax.while_loop(continue_step, step, (d, 0.0, obs, obs_stats_init, 0, False))

  total_reward = carry[1]
  obs_stats = carry[3]

  return total_reward, obs_stats


# jit and vmap rollout
v_rollout = jax.jit(jax.vmap(rollout, in_axes=(None, 0, 0, None, None)))
v_rollout_eval = jax.jit(jax.vmap(rollout, in_axes=(None, None, 0, None, None)))
