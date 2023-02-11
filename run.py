import collections
import functools
import os
import pickle
import random
import time

import gym
import numpy as np
import scipy
import torch

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

gym.logger.set_level(gym.logger.ERROR)

from atari_data import get_human_normalized_score
from atari_preprocessing import AtariPreprocessing

# --- Setup
seed = 100
random.seed(seed)
np.random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
torch.manual_seed(seed)

# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hide GPU from tf, since tf.io.encode_jpeg/decode_jpeg seem to cause GPU memory leak.
tf.config.set_visible_devices([], "GPU")

# --- Create environments
class SequenceEnvironmentWrapper(gym.Wrapper):
    def __init__(self, env, num_stack_frames: int = 1, jpeg_obs: bool = False):
        super().__init__(env)
        self.num_stack_frames = num_stack_frames
        self.jpeg_obs = jpeg_obs

        self.obs_stack = collections.deque([], maxlen=self.num_stack_frames)
        self.act_stack = collections.deque([], maxlen=self.num_stack_frames)
        self.rew_stack = collections.deque([], maxlen=self.num_stack_frames)
        self.done_stack = collections.deque([], maxlen=self.num_stack_frames)
        self.info_stack = collections.deque([], maxlen=self.num_stack_frames)

    @property
    def observation_space(self):
        parent_obs_space = self.env.observation_space
        act_space = self.env.action_space
        episode_history = {
            "observations": gym.spaces.Box(
                np.stack([parent_obs_space.low] * self.num_stack_frames, axis=0),
                np.stack([parent_obs_space.high] * self.num_stack_frames, axis=0),
                dtype=parent_obs_space.dtype,
            ),
            "actions": gym.spaces.Box(0, act_space.n, [self.num_stack_frames], dtype=act_space.dtype),
            "rewards": gym.spaces.Box(-np.inf, np.inf, [self.num_stack_frames]),
        }
        return gym.spaces.Dict(**episode_history)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if self.jpeg_obs:
            obs = self._process_jpeg(obs)

        # Create a N-1 "done" past frames.
        self.pad_current_episode(obs, self.num_stack_frames - 1)
        # Create current frame (but with placeholder actions and rewards).
        self.obs_stack.append(obs)
        self.act_stack.append(0)
        self.rew_stack.append(0)
        self.done_stack.append(0)
        self.info_stack.append(None)
        return self._get_obs()

    def step(self, action: np.ndarray):
        """Replaces env observation with fixed length observation history."""
        # Update applied action to the previous timestep.
        self.act_stack[-1] = action
        obs, rew, done, info = self.env.step(action)
        if self.jpeg_obs:
            obs = self._process_jpeg(obs)
        self.rew_stack[-1] = rew
        # Update frame stack.
        self.obs_stack.append(obs)
        self.act_stack.append(0)  # Append unknown action to current timestep.
        self.rew_stack.append(0)
        self.info_stack.append(info)
        return self._get_obs(), rew, done, info

    def pad_current_episode(self, obs, n):
        # Prepad current episode with n steps.
        for _ in range(n):
            self.obs_stack.append(np.zeros_like(obs))
            self.act_stack.append(0)
            self.rew_stack.append(0)
            self.done_stack.append(1)
            self.info_stack.append(None)

    def _process_jpeg(self, obs):
        obs = np.expand_dims(obs, axis=-1)  # tf expects channel-last
        obs = tf.io.decode_jpeg(tf.io.encode_jpeg(obs))
        obs = np.array(obs).transpose(2, 0, 1)  # to channel-first
        return obs

    def _get_obs(self):
        r"""Return current episode's N-stacked observation.

        For N=3, the first observation of the episode (reset) looks like:

        *= hasn't happened yet.

        GOAL  OBS  ACT  REW  DONE
        =========================
        g0    0    0.   0.   True
        g0    0    0.   0.   True
        g0    x0   0.   0.   False

        After the first step(a0) taken, yielding x1, r0, done0, info0, the next
        observation looks like:

        GOAL  OBS  ACT  REW  DONE
        =========================
        g0    0    0.   0.   True
        g0    x0   0.   0.   False
        g1    x1   a0   r0   d0

        A more chronologically intuitive way to re-order the column data would be:

        PREV_ACT  PREV_REW  PREV_DONE CURR_GOAL CURR_OBS
        ================================================
        0.        0.        True      g0        0
        0.        0.        False*    g0        x0
        a0        r0        info0     g1        x1

        Returns:
        episode_history: np.ndarray of observation.
        """
        episode_history = {
            "observations": np.stack(self.obs_stack, axis=0),
            "actions": np.stack(self.act_stack, axis=0),
            "rewards": np.stack(self.rew_stack, axis=0),
        }
        return episode_history


# from https://github.com/facebookresearch/moolib/blob/06e7a3e80c9f52729b4a6159f3fb4fc78986c98e/examples/atari/environment.py
def create_env(env_name, sticky_actions=False, noop_max=30, terminal_on_life_loss=False):
    env = gym.make(  # Cf. https://brosa.ca/blog/ale-release-v0.7
        f"ALE/{env_name}-v5",
        obs_type="grayscale",  # "ram", "rgb", or "grayscale".
        frameskip=1,  # Action repeats. Done in wrapper b/c of noops.
        repeat_action_probability=0.25 if sticky_actions else 0.0,  # Sticky actions.
        max_episode_steps=108000 // 4,
        full_action_space=True,  # Use all actions.
        render_mode=None,  # None, "human", or "rgb_array".
    )

    # Using wrapper from seed_rl in order to do random no-ops _before_ frameskipping.
    # gym.wrappers.AtariPreprocessing doesn't play well with the -v5 versions of the game.
    env = AtariPreprocessing(
        env,
        frame_skip=4,
        terminal_on_life_loss=terminal_on_life_loss,
        screen_size=84,
        max_random_noops=noop_max,  # Max no-ops to apply at the beginning.
    )
    # env = gym.wrappers.FrameStack(env, num_stack=4)  # frame stack done separately
    env = SequenceEnvironmentWrapper(env, num_stack_frames=4, jpeg_obs=True)
    return env


env_name = "Breakout"
num_envs = 8
env_fn = lambda: create_env(env_name)
envs = [env_fn() for _ in range(num_envs)]
print(f"num_envs: {num_envs}", envs[0])

# --- Create offline RL dataset

# --- Create model
from multigame_dt import MultiGameDecisionTransformer

OBSERVATION_SHAPE = (84, 84)
PATCH_SHAPE = (14, 14)
NUM_ACTIONS = 18  # Maximum number of actions in the full dataset.
# rew=0: no reward, rew=1: score a point, rew=2: end game rew=3: lose a point
NUM_REWARDS = 4
RETURN_RANGE = [-20, 100]  # A reasonable range of returns identified in the dataset

model = MultiGameDecisionTransformer(
    img_size=OBSERVATION_SHAPE,
    patch_size=PATCH_SHAPE,
    num_actions=NUM_ACTIONS,
    num_rewards=NUM_REWARDS,
    return_range=RETURN_RANGE,
    d_model=1280,
    num_layers=10,
    dropout_rate=0.1,
    predict_reward=True,
    single_return_token=True,
    conv_dim=256,
)
print(model)

# --- Load pretrained weights
from load_pretrained import load_jax_weights

model_params, model_state = pickle.load(open("checkpoint_38274228.pkl", "rb"))
load_jax_weights(model, model_params)
model = model.to(device=device)

# --- Train model
model.train()

# --- Save/Load model weights
# torch.save(model.state_dict(), "model.pth")
# model.load_state_dict(torch.load("model.pth"))

# --- Evaluate model
def _batch_rollout(envs, policy_fn, num_episodes, log_interval=None):
    r"""Roll out a batch of environments under a given policy function."""
    num_batch = len(envs)
    num_steps = envs[0].spec.max_episode_steps
    assert num_episodes % num_batch == 0

    rng = torch.Generator()
    seeds_list = [random.randint(0, 2**32 - 1) for _ in range(num_episodes)]
    print(f"seeds: {seeds_list}")

    rew_sum_list = []
    for c in range(num_episodes // num_batch):
        seeds = seeds_list[c * num_batch : (c + 1) * num_batch]
        rng.manual_seed(seeds[0])

        obs_list = [env.reset(seed=seeds[i]) for i, env in enumerate(envs)]
        obs = {k: np.stack([obs[k] for obs in obs_list], axis=0) for k in obs_list[0]}
        rew_sum = np.zeros(num_batch, dtype=np.float32)
        done = np.zeros(num_batch, dtype=np.int32)
        start = time.perf_counter()
        for t in range(num_steps):
            done_prev = done
            obs = {k: torch.tensor(v, device=device) for k, v in obs.items()}
            actions = policy_fn(obs, rng=rng, deterministic=False)

            # Collect step results and stack as a batch.
            step_results = [env.step(act) for env, act in zip(envs, actions.cpu().numpy())]
            obs_list = [result[0] for result in step_results]
            obs = {k: np.stack([obs[k] for obs in obs_list], axis=0) for k in obs_list[0]}
            rew = np.stack([result[1] for result in step_results])
            done = np.stack([result[2] for result in step_results])

            done = np.logical_or(done, done_prev).astype(np.int32)
            rew = rew * (1 - done)
            rew_sum += rew

            if log_interval and t % log_interval == 0:
                elapsed = time.perf_counter() - start
                print(f"step: {t}, fps: {(num_batch * t / elapsed):.2f}, done: {done.astype(np.int32)}, rew_sum: {rew_sum}")

            # Don't continue if all environments are done.
            if np.all(done):
                break

        rew_sum_list.append(rew_sum)
    return np.concatenate(rew_sum_list)


model.eval()
optimal_action_fn = functools.partial(
    model.optimal_action,
    return_range=RETURN_RANGE,
    single_return_token=True,
    opt_weight=0,
    num_samples=128,
    action_temperature=1.0,
    return_temperature=0.75,
    action_top_percentile=50,
    return_top_percentile=None,
)

task_results = {}
task_results["rew_sum"] = _batch_rollout(envs, optimal_action_fn, num_episodes=16, log_interval=100)
[env.close() for env in envs]

# --- Log metrics
def print_metrics(metric):
    print(f"mean: {np.mean(metric):.2f}")
    print(f"std: {np.std(metric):.2f}")
    print(f"median: {np.median(metric):.2f}")
    print(f"iqm: {scipy.stats.trim_mean(metric, proportiontocut=0.25):.2f}")


print("rew_sum")
print_metrics(task_results["rew_sum"])

print("-" * 10)

task_results["human_normalized_score"] = [
    get_human_normalized_score(env_name.lower(), score) for score in task_results["rew_sum"]
]
print("human_normalized_score")
print_metrics(task_results["human_normalized_score"])
