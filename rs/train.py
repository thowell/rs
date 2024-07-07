from rs.search import initialize_search, eval_search
from rs.utilities import load_policy, parse

# import environments
from rs.envs.ant import ant_environment
from rs.envs.cheetah import cheetah_environment
from rs.envs.humanoid import humanoid_environment
from rs.envs.walker import walker_environment


def train_settings(env, settings):
  """Print training information."""
  print("Settings:")
  print(f" environment: {env}")
  print(f"  nsample: {settings['nsample']} | ntop: {settings['ntop']}")
  print(f"  niter: {settings['niter']} | neval: {settings['neval']}")
  print(
      f"  nhorizon_search: {settings['nhorizon_search']} | nhorizon_eval: {settings['nhorizon_eval']}"
  )
  print(
      f"  random_step: {settings['random_step']} | update_step: {settings['update_step']}"
  )
  print(f"  nenveval: {settings['nenveval']}")
  print(f"  reward_shift: {settings['reward_shift']}")


def train():
  """Train linear policy with random search."""
  # parse settings
  args = parse()

  # environment
  env, p, settings, vis = eval(str(args.env + "_environment()"))

  # load policy
  if args.load != "":
    p = load_policy(p, args.load)

  # search
  if args.search:
    # update settings with parsed arguments
    for k, v in vars(args).items():
      if k in settings and v is not None:
        settings[k] = v

    # settings
    train_settings(args.env, settings)

    # initialize
    s = initialize_search(**settings)

    # search + evaluation
    p = eval_search(s, env, p, seed=args.seed, checkpoint=args.checkpoint)

  # visualize policy
  if args.visualize:
    vis(p)


if __name__ == "__main__":
  train()
