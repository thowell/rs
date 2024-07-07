from functools import partial
from typing import Tuple
import jax
import jax.numpy as jnp
from mujoco.mjx._src import dataclasses
import pathlib
from pathlib import Path
import pickle
import time

from rs.environment import Environment, v_rollout, v_rollout_eval
from rs.policy import Policy, v_noisy_policy

# running statistics: (mean, var, count)
RunningStatistics = Tuple[jax.Array, jax.Array, int]


class Search(dataclasses.PyTreeNode):
  """Search settings.

  Attributes:
    nsample: number of samples to evaluate at each iteration (+, -)
    ntop: number of top samples used for update
    niter: number of iterations
    neval: number of evaluations
    nhorizon_search: number of steps to simulate policy for improvement with search
    nhorizon_eval: number of steps to simulate policy for evaluation
    random_step: search size; p +- step * N(0, I)
    update_step: parameter update size: W += step * update
    step_direction: directions for random_step
    nenveval: number of environments for evaluation
    reward_shift: value to subtract from per-timestep environment reward
  """

  nsample: int
  ntop: int
  niter: int
  neval: int
  nhorizon_search: int
  nhorizon_eval: int
  random_step: float
  update_step: float
  step_direction: jax.Array
  nenveval: int
  reward_shift: float


def initialize_search(
    nsample: int,
    ntop: int,
    niter: int,
    neval: int,
    nhorizon_search: int,
    nhorizon_eval: int,
    random_step: float,
    update_step: float,
    nenveval: int = 128,
    reward_shift: float = 0.0
) -> Search:
  """Create Search with settings.

  Args:
      nsample (int)
      ntop (int)
      niter (int)
      neval (int)
      nhorizon_search (int)
      nhorizon_eval (int)
      random_step (float)
      update_step (float)
      nenveval (int)
      reward_shift (float)

  Returns:
      Search
  """
  # step direction
  step_direction = jnp.concatenate(
      [
          random_step * jnp.ones(nsample),
          -random_step * jnp.ones(nsample),
      ]
  )

  return Search(
      nsample=nsample,
      ntop=ntop,
      niter=niter,
      neval=neval,
      nhorizon_search=nhorizon_search,
      nhorizon_eval=nhorizon_eval,
      random_step=random_step,
      update_step=update_step,
      step_direction=step_direction,
      nenveval=nenveval,
      reward_shift=reward_shift,
  )


@partial(jax.jit, static_argnums=(5,))
def search(
    s: Search,
    env: Environment,
    p: Policy,
    obs_stats: RunningStatistics,
    rng: jax.Array,
    iter: int = 1,
) -> Tuple[Policy, RunningStatistics, Tuple[jax.Array, jax.Array]]:
  """Improve policy with random search.

  Returns:
    Policy
    obs_stats (RunningStatistics)
    reward_stats (Tuple[jax.Array, jax.Array, jax.Array]): reward statistics  (mean, std, count)
  """

  ## iteration
  def iteration(carry, _):
    # unpack
    rng, p, obs_stats = carry

    # random
    keys = jax.random.split(rng, 3 * s.nsample + 1)
    rng = keys[-1]
    key_perturb = keys[:s.nsample]
    key_policy_positive_negative = jnp.concatenate([key_perturb, key_perturb])

    # noisy policies (and perturbations)
    policy_noisy, policy_perturb = v_noisy_policy(
        p, s.step_direction, key_policy_positive_negative
    )

    # reset
    key_reset = keys[s.nsample:-1]
    d_random = jax.vmap(env.reset, in_axes=(None, None, 0))(
        env.model, env.data, key_reset
    )

    # rollout noisy policies
    rewards, obs_stats_rollout = v_rollout(env, policy_noisy, d_random, s.reward_shift, s.nhorizon_search)

    # collect running statistics from environments
    def merge_running_statistics(stats0, stats1):
      # https://github.com/a-mitani/welford/blob/b7f96b9ad5e803d6de665c7df1cdcfb2a53bddc8/welford/welford.py#L132
      mean0, var0, count0 = stats0
      mean1, var1, count1 = stats1

      count = count0 + count1

      delta = mean0 - mean1
      delta2 = delta * delta

      mean = (count0 * mean0 + count1 * mean1) / count
      var = var0 + var1 + delta2 * count0 * count1 / count

      return (mean, var, count), count1

    obs_stats_updated, counts = jax.lax.scan(
        merge_running_statistics, obs_stats, obs_stats_rollout
    )

    # collect reward pairs
    rewards_pos_neg = jnp.vstack([rewards[: s.nsample], rewards[s.nsample :]])

    # reward pair max
    rewards_pos_neg_max = jnp.max(rewards_pos_neg, axis=0)

    # sort reward pairs descending, keep first ntop
    sort = jnp.argsort(rewards_pos_neg_max, descending=True)[: s.ntop]

    # ntop best pairs
    rewards_best = rewards_pos_neg[:, sort]

    # best pair statistics
    rewards_best_mean = rewards_best.flatten().mean()
    rewards_best_std = rewards_best.flatten().std()
    rewards_best_std = jnp.where(rewards_best_std < 1.0e-7, float("inf"), rewards_best_std)

    # new weights
    # https://arxiv.org/pdf/1803.07055.pdf: algorithm 2
    weight_update = jnp.einsum(
        "i,ijk->jk",
        jnp.dot(jnp.array([1.0, -1.0]), rewards_best),
        policy_perturb[sort],
    )
    weight = (
        p.weight
        + s.update_step / s.ntop / (rewards_best_std + 1.0e-5) * weight_update
    )

    # update policy
    mean, var, count = obs_stats_updated
    std = jnp.where(count > 1, jnp.sqrt(var / (count - 1)), 1.0)
    std = jnp.where(std < 1.0e-7, float("inf"), std)
    p = p.replace(
        weight=weight,
        shift=mean,
        scale=std,
    )

    return (rng, p, obs_stats_updated), (
        rewards_best_mean,
        rewards_best_std,
        jnp.mean(counts),
    )

  # loop
  initial_count = obs_stats[2]
  carry, reward_stats = jax.lax.scan(
      iteration,
      (
          rng,
          p,
          obs_stats,
      ),
      None,
      length=iter,
  )
  policy = carry[1]
  obs_stats_update = carry[2]
  env_steps = obs_stats_update[2] - initial_count
  return policy, obs_stats_update, reward_stats, env_steps


def eval_search(
    s: Search,
    env: Environment,
    p: Policy,
    seed: int = 0,
    checkpoint: str = None,
) -> Policy:
  """Improve policy with random search and provide evaluation information during training.

  Returns
    Policy
  """
  print("Search:")
  
  # vmap reset
  v_reset = jax.jit(jax.vmap(env.reset, in_axes=(None, None, 0)))

  # create logs directory for checkpoints
  if checkpoint is not None:
    # directory
    checkpoint_dir = pathlib.Path(__file__).parent / "checkpoint"
    
    # create directory
    Path(str(checkpoint_dir)).mkdir(parents=True, exist_ok=True)
    
    # create subdirectory
    Path(str(checkpoint_dir) + checkpoint).mkdir(parents=True, exist_ok=True)
    

  # start total timer
  start_total = time.time()

  # initialize key from seed
  key = jax.random.PRNGKey(seed)

  # initialize observation statistics
  obs_stats = (
      jnp.zeros(env.nobservation),
      jnp.ones(env.nobservation),
      jnp.ones(1).astype(int),
  )

  # evaluation iterations
  iter_per_eval = int(s.niter / s.neval)
  for i in range(s.neval):
    # random
    rng, key = jax.random.split(key)

    # search
    start = time.time()
    p, obs_stats, _, search_steps = search(
        s, env, p, obs_stats, rng, iter=iter_per_eval
    )
    
    # stop search timer
    search_time = time.time() - start
    
    ## evaluate
    # random
    keys = jax.random.split(rng, s.nenveval + 2)
    key_eval_rollout = keys[:-2]
    rng, key = keys[-2:]

    # environment reset
    d_random = v_reset(env.model, env.data, key_eval_rollout)
        
    # rollout current
    rewards_eval, obs_stats_eval = v_rollout_eval(env, p, d_random, 0.0, s.nhorizon_eval)
    
    # stats
    reward_mean = rewards_eval.mean()
    reward_std = rewards_eval.std()
    avg_episode_len = int(obs_stats_eval[2].mean()) - 1
    env_steps = int(obs_stats[2][0]) - 1
    print(
        f"iteration ({(i + 1) * iter_per_eval} / {s.niter}): reward = {reward_mean:.2f} +- {reward_std:.2f} | time = {search_time:.2f} | avg episode length: {avg_episode_len} / {s.nhorizon_eval} | global steps: {env_steps} | steps/second: {int(search_steps[0]/search_time)}"
    )

    # save checkpoint
    if checkpoint is not None:
      checkpoint_path = (
          str(checkpoint_dir)
          + "/"
          + checkpoint
          + "/"
          + checkpoint
          + "_"
          + str(i)
          + "_"
          + "{:.2f}".format(reward_mean)
          + "_"
          + "{:.2f}".format(reward_std)
      )
      with open(checkpoint_path, "wb") as file:
        pickle.dump(p, file)

  # total time
  print(f"\ntotal time: {time.time() - start_total:.2f}")

  return p
