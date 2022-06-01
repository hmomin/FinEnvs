# @title Import Brax and some helper modules

import brax
import functools
import gym
import jax
import time
import torch
from brax import envs
from brax import jumpy as jp
from brax.envs import to_torch
from brax.io import html
from brax.io import image
from jax import numpy as jnp
from IPython.display import HTML, Image
from time import time

v = torch.ones(1, device="cuda")  # init torch cuda before jax

# @title Visualizing pre-included Brax environments { run: "auto" }
# @markdown Select an environment to preview it below:

environment = "ant"  # @param ['ant', 'halfcheetah', 'hopper', 'humanoid', 'reacher', 'walker2d', 'fetch', 'grasp', 'ur5e']
env = envs.create(env_name=environment)
state = env.reset(rng=jp.random_prngkey(seed=0))

# HTML(html.render(env.sys, [state.qp]))

start = time()
rollout = []
for i in range(100):
    # wiggle sinusoidally with a phase shift per actuator
    action = jp.sin(i * jp.pi / 15 + jp.arange(0, env.action_size) * jp.pi)
    state = env.step(state, action)
    rollout.append(state)
print(f"Elapsed time: {time() - start}s")

start = time()
# jit compile env.step:
state = jax.jit(env.step)(state, jnp.ones((env.action_size,)))

for _ in range(100):
    state = jax.jit(env.step)(state, jnp.ones((env.action_size,)))
print(f"Elapsed time: {time() - start}s")

# HTML(html.render(env.sys, [s.qp for s in rollout]))

# Image(image.render(env.sys, [s.qp for s in rollout], width=320, height=240))

entry_point = functools.partial(envs.create_gym_env, env_name="ant")
if "brax-ant-v0" not in gym.envs.registry.env_specs:
    gym.register("brax-ant-v0", entry_point=entry_point)

# create a gym environment that contains 4096 parallel ant environments
gym_env = gym.make("brax-ant-v0", batch_size=4096)

# wrap it to interoperate with torch data structures
gym_env = to_torch.JaxToTorchWrapper(gym_env, device="cuda")

# jit compile env.reset
obs = gym_env.reset()

# jit compile env.step
action = torch.rand(gym_env.action_space.shape, device="cuda") * 2 - 1
obs, reward, done, info = gym_env.step(action)

start = time()

for _ in range(100):
    action = torch.rand(gym_env.action_space.shape, device="cuda") * 2 - 1
    obs, rewards, done, info = gym_env.step(action)

duration = time() - start
print(
    f"time for {409_600} steps: {duration:.2f}s ({int(409_600 / duration)} steps/sec)"
)
