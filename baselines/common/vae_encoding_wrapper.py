import gym
import numpy as np
from gym import spaces

class VAEEncodingWrapper(gym.Wrapper):
    def __init__(self, env, vae):
        super().__init__(env)
        self.vae = vae
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(vae.latent_dim,))
        self.obs_buffer = []

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.obs_buffer.append(obs)
        if len(self.obs_buffer) > 4:
            self.obs_buffer = self.obs_buffer[-4:]
        obs_timespan = np.concatenate(self.obs_buffer[-4:], axis=-1)
        obs = self.vae.encode(obs_timespan[None] / 255)[0]
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset()
        self.obs_buffer = []
        self.obs_buffer.extend([np.zeros_like(obs)] * 3)
        self.obs_buffer.append(obs)
        obs_timespan = np.concatenate(self.obs_buffer[-4:], axis=-1)
        obs = self.vae.encode(obs_timespan[None] / 255)[0]
        return obs