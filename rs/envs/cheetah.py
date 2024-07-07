import jax
import jax.numpy as jnp
from mujoco import mjx

from rs.policy import initialize_policy
from rs.environment import Environment
from rs.utilities import load_model, visualize


def cheetah_environment():
  """Return Half Cheetah learning environment."""
  # load MuJoCo model + data for MuJoCo C | MuJoCo XLA
  mc, dc, mx, dx = load_model("cheetah")

  # sizes
  naction = mx.nu
  nobservation = mx.nq - 1 + mx.nv
  ndecimation = 5

  def reward(m: mjx.Model, d0: mjx.Data, d1: mjx.Data, dt: float) -> float:
    # https://github.com/openai/gym/blob/master/gym/envs/mujoco/half_cheetah_v4.py
    r_forward = (d1.qpos[0] - d0.qpos[0]) / dt
    r_control = -jnp.dot(d1.ctrl, d1.ctrl)
    return 1.0 * r_forward + 0.1 * r_control

  def observation(m: mjx.Model, d: mjx.Data) -> jax.Array:
    return jnp.hstack([d.qpos[1:], d.qvel])

  def reset(m: mjx.Model, d: mjx.Data, rng) -> mjx.Data:
    # key
    key_pos, key_vel = jax.random.split(rng)

    # qpos
    is_limited = mx.jnt_limited == 1
    lower, upper = mx.jnt_range[is_limited].T
    qpos = jax.random.uniform(key_pos, shape=(mx.nq,), minval=-0.1, maxval=0.1)
    qclip = jnp.clip(qpos[is_limited], a_min=lower, a_max=upper)
    qpos = qpos.at[is_limited].set(qclip)

    # qvel
    qvel = jnp.clip(
        0.1 * jax.random.normal(key_vel, shape=(mx.nv,)), a_min=-1.0, a_max=1.0
    )

    # update data
    d = d.replace(qpos=qpos, qvel=qvel)

    return d

  def done(m: mjx.Model, d: mjx.Data) -> bool:
    return False

  env = Environment(
      model=mx,
      data=dx,
      reward=reward,
      observation=observation,
      reset=reset,
      done=done,
      naction=naction,
      nobservation=nobservation,
      ndecimation=ndecimation,
  )

  # policy
  limits = (mc.actuator_ctrlrange[:, 0], mc.actuator_ctrlrange[:, 1])
  p = initialize_policy(env.naction, env.nobservation, limits)

  # search settings
  settings = {
      "nsample": 32,
      "ntop": 4,
      "niter": 100,
      "neval": 10,
      "nhorizon_search": 1000,
      "nhorizon_eval": 1000,
      "random_step": 0.03,
      "update_step": 0.02,
      "nenveval": 128,
      "reward_shift": 0.0,
  }

  # tracking camera
  def lookat(viewer, data):
    viewer.cam.lookat[0] = data.qpos[0]
    viewer.cam.lookat[2] = data.qpos[1] + 0.5
    viewer.cam.distance = 4.0

  # visualize
  def vis(p):
    visualize(mc, dc, env, p, lookat=lookat)

  return env, p, settings, vis


if __name__ == "__main__":
  # environment
  env, p, settings, vis = cheetah_environment()

  # visualize
  vis(p)
