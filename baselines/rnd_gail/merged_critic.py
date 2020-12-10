import numpy as np
import gym
from tensorflow.python.ops.gen_math_ops import exp
from .rnd_critic import RND_Critic_CNN, RND_Critic, Enc_Critic
from .mmd_critic import MMD_Critic

def make_critic(env, exp_data, hid_size=128, rnd_hid_size=128, reward_type=0, scale=250000, CNN_critic=False, rnd_cnn_type=1):
    if type(env.action_space) is gym.spaces.Discrete:
        ac_size = env.action_space.n
    else:
        ac_size = env.action_space.sample().shape[0]
    if CNN_critic:
        ob_size = exp_data[0].shape[1:]
    else:
        ob_size = env.observation_space.shape[0]
    if reward_type == 0:
        if CNN_critic:
            critic = RND_Critic_CNN(ob_size, ac_size, hid_size=hid_size, rnd_hid_size=rnd_hid_size, scale=scale, rnd_hid_layer=1, hid_layer=1, rnd_cnn_type=rnd_cnn_type)
        else:
            critic = RND_Critic(ob_size, ac_size, hid_size=hid_size, rnd_hid_size=rnd_hid_size, scale=scale, rnd_hid_layer=1)
    elif reward_type == 1:
        critic = Enc_Critic(ob_size, ac_size, hid_size=hid_size, scale=scale)
    else:
        merged_sa = np.concatenate(exp_data, axis=1)
        critic = MMD_Critic(ob_size, ac_size, merged_sa)
    return critic