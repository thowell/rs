import jax
import jax.numpy as jnp
from mujoco import mjx

from rs.policy import initialize_policy
from rs.environment import Environment
from rs.utilities import load_model, visualize


def humanoid_environment():
  # load MuJoCo model + data for MuJoCo C | MuJoCo XLA
  mc, dc, mx, dx = load_model("humanoid")

  # sizes
  naction = mx.nu
  nobservation = mx.nq - 2 + mx.nv
  nobservation += (mx.nbody - 1) * 10 # d.cinert
  nobservation += (mx.nbody - 1) * 6  # d.cvel
  nobservation += mx.nv - 6 # d.qfrc_actuator
  ndecimation = 5

  # healthy
  def is_healthy(m: mjx.Model, d: mjx.Data) -> bool:
    z = d.qpos[2]

    min_z, max_z = 1.0, 2.0

    healthy_z = jnp.logical_and(min_z < z, z < max_z)

    return healthy_z


  def reward(m: mjx.Model, d0: mjx.Data, d1: mjx.Data, dt: float) -> float:
    # https://github.com/openai/gym/blob/master/gym/envs/mujoco/walker2d_v4.py

    # healthy
    r_healthy = is_healthy(m, d1).astype(float)

    # forward
    r_forward = (d1.qpos[0] - d0.qpos[0]) / dt

    # control penalty
    r_control = jnp.sum(jnp.square(d1.ctrl))

    return 5.0 * r_healthy + 1.25 * r_forward - 0.1 * r_control


  def observation(m: mjx.Model, d: mjx.Data) -> jax.Array:
    pos = d.qpos[2:]
    vel = d.qvel

    com_inertia = d.cinert[1:].flatten()
    com_velocity = d.cvel[1:].flatten()
    actuator_forces = d.qfrc_actuator[6:]

    return jnp.hstack([pos, vel, com_inertia, com_velocity, actuator_forces])


  def reset(m: mjx.Model, d: mjx.Data, rng) -> mjx.Data:
    # key
    key_pos, key_vel = jax.random.split(rng)

    # qpos
    qpos = m.qpos0 + 0.01 * jax.random.uniform(
        key_pos, shape=(m.nq,), minval=-1.0, maxval=1.0
    )

    # qvel
    qvel = 0.01 * jax.random.uniform(
        key_vel,
        shape=(m.nv,), minval=-1.0, maxval=1.0,
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
        "nsample": 320,
        "ntop": 320,
        "niter": 1000,
        "neval": 100,
        "nhorizon_search": 1000,
        "nhorizon_eval": 1000,
        "random_step": 0.0075,
        "update_step": 0.02,
        "nenveval": 128,
        "reward_shift": 5.0,
  }
  
  # tracking camera
  def lookat(viewer, data):
    viewer.cam.lookat[0] = data.qpos[0]
    viewer.cam.lookat[1] = data.qpos[1]
    viewer.cam.lookat[2] = data.qpos[2]
    viewer.cam.distance = 4.0

  # visualize
  def vis(p):
    visualize(mc, dc, env, p, lookat=lookat)

  return env, p, settings, vis


if __name__ == "__main__":
  # environment
  env, p, settings, vis = humanoid_environment()

  # visualize
  vis(p)
