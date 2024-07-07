from typing import Callable, Tuple
import argparse
import mujoco
import mujoco.viewer
from mujoco import mjx
import pathlib
import pickle
import time

from rs.environment import Environment
from rs.policy import Policy, policy

def load_model(model: str) -> Tuple[mjx.Model, mjx.Data]:
  """Load MuJoCo XLA model and data."""
  # path to model
  path = pathlib.Path(__file__).parent / str("models/" + model + ".xml")

  # mjc model + data
  m = mujoco.MjModel.from_xml_path(str(path))
  d = mujoco.MjData(m)

  # mjx model + data
  m_mjx = mjx.put_model(m)
  d_mjx = mjx.put_data(m, d)

  return m, d, m_mjx, d_mjx


def load_policy(p: Policy, file: str) -> Policy:
  """Load policy from checkpoint."""
  checkpoint_dir = pathlib.Path(__file__).parent / "checkpoint"
  checkpoint_path = str(checkpoint_dir) + "/" + file
  try:
    with open(checkpoint_path, 'rb') as f:
      p = pickle.load(f)
      
    print(f"Success: load policy from: {checkpoint_path}")
  except Exception as e:
    print(f"Failure: load policy from: {checkpoint_path}")
  
  return p


def visualize(
    m: mujoco.MjModel,
    d: mujoco.MjData,
    env: Environment,
    p: Policy,
    lookat: Callable = lambda viewer, data: None,
):
  """Visualize learned policy with MuJoCo passive viewer."""
  # visualize policy
  with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
      # start timer
      t0 = time.time()

      # observations
      obs = env.observation(m, d)

      # compute and set actions
      d.ctrl = policy(p, obs)

      # simulate
      for _ in range(env.ndecimation):
        # step physics
        mujoco.mj_step(m, d)

        # camera tracking
        lookat(viewer, d)

        # sync visualization
        viewer.sync()

      # wait
      elapsed = time.time() - t0
      time.sleep(max(m.opt.timestep * env.ndecimation - elapsed, 0.0))


def parse():
  """Parse command line arguments for environment and search settings."""
  # parser
  parser = argparse.ArgumentParser()

  # search setup
  parser.add_argument("--env", type=str, choices=["ant", "cheetah", "humanoid", "walker"], default="cheetah", help="Learning environment (default: cheetah)")
  parser.add_argument("--search", action="store_true", help="Random search to find policy")
  parser.add_argument("--load", type=str, default="", help="Path to saved policy (default: "")")
  parser.add_argument("--checkpoint", type=str, default=None, help="Path to saved policy (default: None)")
  parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
  parser.add_argument("--visualize", action="store_true", help="Visualize policy")

  # search settings
  parser.add_argument("--nsample", type=int, default=None, help="Number of random directions to sample")
  parser.add_argument("--ntop", type=int, default=None, help="Number of random directions to use for policy update")
  parser.add_argument("--niter", type=int, default=None, help="Number of policy updates")
  parser.add_argument("--neval", type=int, default=None, help="Number of policy evaluations during search")
  parser.add_argument("--nhorizon_search", type=int, default=None, help="Number of environment steps during policy search")
  parser.add_argument("--nhorizon_eval", type=int, default=None, help="Number of environment steps during policy evaluation")
  parser.add_argument("--random_step", type=float, default=None, help="Step size for random direction")
  parser.add_argument("--update_step", type=float, default=None, help="Step size for policy update")
  parser.add_argument("--nenveval", type=int, default=None, help="Number of environments for policy evaluation")
  parser.add_argument("--reward_shift", type=float, default=None, help="Subtract from per-timestep reward")
  
  # args
  args = parser.parse_args()
  
  return args
