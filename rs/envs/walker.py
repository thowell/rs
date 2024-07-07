import jax
import jax.numpy as jnp
from mujoco import mjx

from rs.policy import initialize_policy
from rs.environment import Environment
from rs.utilities import load_model, visualize

def walker_environment():
  # load MuJoCo model + data for MuJoCo C | MuJoCo XLA
  mc, dc, mx, dx = load_model("walker")

  # sizes
  naction = mx.nu
  nobservation = mx.nq - 1 + mx.nv
  ndecimation = 4
  
  # healthy
  def is_healthy(m: mjx.Model, d: mjx.Data) -> bool:
    z, angle = d.qpos[1:3]

    min_z, max_z = 0.8, 2.0
    min_angle, max_angle = -1.0, 1.0

    healthy_z = jnp.logical_and(min_z < z, z < max_z)
    healthy_angle = jnp.logical_and(min_angle < angle, angle < max_angle)

    return jnp.logical_and(healthy_z, healthy_angle)


  def reward(m: mjx.Model, d0: mjx.Data, d1: mjx.Data, dt: float) -> float:
    # https://github.com/openai/gym/blob/master/gym/envs/mujoco/walker2d_v4.py

    # healthy
    r_healthy = is_healthy(m, d1).astype(float)

    # forward
    r_forward = (d1.qpos[0] - d0.qpos[0]) / dt

    # control penalty
    r_control = -jnp.sum(jnp.square(d1.ctrl))

    return 1.0 * r_healthy + 1.0 * r_forward + 0.001 * r_control


  def observation(m: mjx.Model, d: mjx.Data) -> jax.Array:
    return jnp.hstack([d.qpos[1:], jnp.clip(d.qvel, -10.0, 10.0)])


  def reset(m: mjx.Model, d: mjx.Data, rng) -> mjx.Data:
    # key
    key_pos, key_vel = jax.random.split(rng)

    # qpos
    qpos = m.qpos0 + 5.0e-3 * jax.random.uniform(
        key_pos, shape=(m.nq,), minval=-1.0, maxval=1.0
    )

    # qvel
    qvel = jnp.zeros(m.nv) + 5.0e-3 * jax.random.uniform(
        key_vel, shape=(m.nv,), minval=-1.0, maxval=1.0
    )

    # update data
    d = d.replace(qpos=qpos, qvel=qvel)

    return d


  def done(m: mjx.Model, d: mjx.Data) -> bool:
    return jnp.where(is_healthy(m, d), 0, 1)


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
        "nsample": 40,
        "ntop": 30,
        "niter": 1000,
        "neval": 100,
        "nhorizon_search": 1000,
        "nhorizon_eval": 1000,
        "random_step": 0.025,
        "update_step": 0.015,
        "nenveval": 128,
        "reward_shift": 1.0}
  
  # tracking camera
  def lookat(viewer, data):
    viewer.cam.lookat[0] = data.qpos[0]
    viewer.cam.lookat[2] = data.qpos[1]
    viewer.cam.distance = 4.0

  # visualize
  def vis(p):
    visualize(mc, dc, env, p, lookat=lookat)

  return env, p, settings, vis

if __name__ == "__main__":
  # environment
  env, p, settings, vis = walker_environment()
  
  # visualize
  vis(p)
