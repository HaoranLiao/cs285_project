'''
Disclaimer: this code is highly based on trpo_mpi at @openai/baselines and @openai/imitation
'''

import argparse
import os.path as osp
import logging
from IPython.lib.security import passwd
from mpi4py import MPI
from tqdm import tqdm

import numpy as np
import gym

import sys
sys.path.append("../../")

from baselines.rnd_gail import cnn_policy, mlp_policy
from baselines.common import set_global_seeds, tf_util as U
from baselines.common.misc_util import boolean_flag, one_hot_encoding
from baselines import bench
from baselines import logger
from baselines.rnd_gail.merged_critic import make_critic

import pickle

DATASET_PATH = "/state/partition1/home/haoranliao/expert_data"
# DATASET_PATH = "/home/jerry/data2/cs285_final_proj/cs285_project/data/mspacman"
# DATASET_PATH = "/mnt/c/Users/haora/Desktop/expert_data"

def get_exp_data(expert_path):
    with open(expert_path, 'rb') as f:
        data = pickle.loads(f.read())

        data["actions"] = np.squeeze(data["actions"])
        data["observations"] = data["observations"]

        return [data["observations"], data["actions"]]


def get_exp_data_atari(expert_path, ae=None):
    print("Start loading dataset...")
    obs, acs = [], []
    for i in tqdm(range(1)):
        with open(expert_path + f'/expert_data_{i}.pkl', 'rb') as f:
            data = pickle.loads(f.read())

            for i in tqdm(range(len(data)), leave=False):
                obs.append(data[i]['observation'])
                ac = one_hot_encoding(data[i]["action"])
                acs.append(ac)

    obs = np.concatenate(obs)
    if ae is not None:
        batch_size = 500
        obs_encoded = []
        i = 0
        while(i * batch_size < len(obs)):
            obs_encoded.append(ae.encode(obs[i * batch_size: (i + 1) * batch_size] / 255))
            i += 1
        obs = np.concatenate(obs_encoded)
    acs = np.concatenate(acs)
    print("Load dataset complete!")
    return [obs, acs]


Log_dir = osp.expanduser("~/workspace/log/mujoco")
Checkpoint_dir = osp.expanduser("~/workspace/checkpoint/mujoco")


def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of GAIL")
    parser.add_argument('--env_id', help='environment ID', default="Hopper-v2")
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default=Checkpoint_dir)
    parser.add_argument('--log_dir', help='the directory to save log file', default=Log_dir)
    parser.add_argument('--load_model_path', help='if provided, load the model', type=str, default=None)
    # Task
    parser.add_argument('--task', type=str, choices=['train', 'evaluate', 'sample'], default='train')
    # for evaluatation
    boolean_flag(parser, 'stochastic_policy', default=False, help='use stochastic/deterministic policy to evaluate')
    # Optimization Configuration
    parser.add_argument('--g_step', help='number of steps to train policy in each epoch', type=int, default=3)
    parser.add_argument('--d_step', help='number of steps to train discriminator in each epoch', type=int, default=1)
    # Network Configuration (Using MLP Policy)
    boolean_flag(parser, 'use_cnn', default=False, help='use cnn for the policy network and the critic')
    boolean_flag(parser, 'encode_1d_obs', default=False, help='use a (variational) autoencoder to encode 2d observations into 1d state representations')
    parser.add_argument('--policy_hidden_size', type=int, default=100)
    parser.add_argument('--adversary_hidden_size', type=int, default=100)
    # Algorithms Configuration
    parser.add_argument('--max_kl', type=float, default=0.01)
    parser.add_argument('--policy_entcoeff', help='entropy coefficiency of policy', type=float, default=0)
    parser.add_argument('--adversary_entcoeff', help='entropy coefficiency of discriminator', type=float, default=1e-3)
    # Traing Configuration
    parser.add_argument('--num_timesteps', help='number of timesteps per episode', type=int, default=5e6)
    # Behavior Cloning
    boolean_flag(parser, 'pretrained', default=False, help='Use BC to pretrain')
    boolean_flag(parser, 'fixed_var', default=False, help='Fixed policy variance')
    parser.add_argument('--BC_max_iter', help='Max iteration for training BC', type=int, default=20)
    parser.add_argument('--gamma', help='Discount factor', type=float, default=0.97)
    boolean_flag(parser, 'popart', default=True, help='Use popart on V function')
    parser.add_argument('--reward', help='Reward Type', type=int, default=0)
    parser.add_argument('--rnd_cnn_type', help='RND Critic CNN Structure', type=int, default=1)
    parser.add_argument('--policy_cnn_type', help='Agent CNN Structure', type=int, default=1)
    parser.add_argument('--rnd_critic_scale', help='RNN Critic Scale', type=int, default=250000)
    return parser.parse_args()


def get_task_name(args):
    task_name = args.env_id.split("-")[0]
    if args.pretrained:
        task_name += "pretrained."
    task_name +="_gamma%f" % args.gamma
    task_name += "_seed" + str(args.seed)
    task_name += "_reward" + str(args.reward)
    task_name += "_kl" + str(args.max_kl)
    task_name += "_g"+str(args.g_step)

    return task_name

def modify_args(args):
    #task specific parameters
    if args.reward<2:
        rnd_iter = 200
        dyn_norm = False

        if args.env_id == "Reacher-v2":
            rnd_iter = 300
            args.gamma = 0.99

        if args.env_id == "HalfCheetah-v2":
            args.pretrained = True

        if args.env_id == "Walker2d-v2":
            args.fixed_var = False

        if args.env_id == "Ant-v2":
            args.pretrained = True
            args.BC_max_iter = 10
            args.fixed_var = False

        if args.env_id == "MsPacman-v0":
            rnd_iter = 5
        return args, rnd_iter, dyn_norm
    else:
        if args.env_id == "Hopper-v2":
            args.gamma = 0.99
            dyn_norm = False

        if args.env_id == "Reacher-v2":
            dyn_norm = True

        if args.env_id == "HalfCheetah-v2":
            dyn_norm = True

        if args.env_id == "Walker2d-v2":
            args.gamma = 0.99
            dyn_norm = True

        if args.env_id == "Ant-v2":
            args.gamma = 0.99
            dyn_norm = False

        return args, 0, dyn_norm


def main(args):
    set_global_seeds(args.seed)
    env = gym.make(args.env_id)
    if args.env_id == "MsPacman-v0":
        from gym import wrappers
        env = wrappers.Monitor(
                env,
                "/tmp",
                force=True,
                video_callable=False,
            )
        from baselines.common.wrappers import wrap_deepmind
        env = wrap_deepmind(env)
    if args.encode_1d_obs:
        from vae.AE import Autoencoder
        from baselines.common.vae_encoding_wrapper import VAEEncodingWrapper
        ae = Autoencoder((84,84,4), [(64, 8, 2), (128, 6, 3), (128, 4, 2), (128, 3, 1)], [1000,500], 100)
        ae.load_model("../../vae/runs/5e-4decay/model")
        U.update_initialized_parameters()
        env = VAEEncodingWrapper(env, ae)
    env.seed(args.seed)

    # env = bench.Monitor(env, logger.get_dir() and
    #                     osp.join(logger.get_dir(), "monitor.json"))


    gym.logger.setLevel(logging.WARN)

    if args.log_dir != Log_dir:
        log_dir = osp.join(Log_dir, args.log_dir)
        save_dir = osp.join(Checkpoint_dir, args.log_dir)
    else:
        log_dir = Log_dir
        save_dir = Checkpoint_dir

    args, rnd_iter, dyn_norm = modify_args(args)

    logger.log(f"rnd_cnn_type: {args.rnd_cnn_type}")
    logger.log(f"policy_cnn_type: {args.policy_cnn_type}")
    logger.log(f"rnd_critic_scale: {args.rnd_critic_scale}")
    logger.log(f"policy_hidden_size: {args.policy_hidden_size}")

    def policy_fn(name, ob_space, ac_space):
        # return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
        #                             hid_size=args.policy_hidden_size, num_hid_layers=2, popart=args.popart, gaussian_fixed_var=args.fixed_var)
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    hid_size=[150, 50], num_hid_layers=2, popart=args.popart, gaussian_fixed_var=args.fixed_var, activation="relu")

            

    def policy_fn_cnn(name, ob_space, ac_space):
        return cnn_policy.CNNPolicy(name=name, policy_cnn_type=args.policy_cnn_type, ob_space=ob_space, ac_space=ac_space,
                                    hid_size=args.policy_hidden_size, num_hid_layers=2, popart=args.popart, gaussian_fixed_var=args.fixed_var)

    if args.task == 'train':

        if args.env_id == "MsPacman-v0":
            if args.encode_1d_obs:
                exp_data = get_exp_data_atari(DATASET_PATH, ae)
            else:
                exp_data = get_exp_data_atari(DATASET_PATH)
        else:
            exp_data = get_exp_data(osp.join(osp.dirname(osp.realpath(__file__)), "../../data/%s.pkl" % args.env_id))


        task_name = get_task_name(args)

        task_name += "_rndcnn"+str(args.rnd_cnn_type) + "_mlpcnn"+str(args.policy_cnn_type)

        logger.configure(dir=log_dir, log_suffix=task_name, format_strs=["log", "stdout"])
        if args.reward == 0:
            if args.env_id == "Humanoid-v2":
                critic = make_critic(env, exp_data, reward_type=args.reward, scale=2500)
            elif args.env_id == "Reacher-v2":
                    critic = make_critic(env, exp_data, rnd_hid_size=20, hid_size=20, reward_type=args.reward, scale=2500)
            elif args.env_id == "HalfCheetah-v2":
                critic = make_critic(env, exp_data, rnd_hid_size=20, hid_size=20, reward_type=args.reward, scale=25000)
            elif args.env_id == "Ant-v2":
                critic = make_critic(env, exp_data, reward_type=args.reward)
            elif args.env_id == "MsPacman-v0":
                critic = make_critic(env, exp_data, hid_size=128, reward_type=args.reward, scale=args.rnd_critic_scale, CNN_critic=args.use_cnn, rnd_cnn_type=args.rnd_cnn_type)
            else:
                critic = make_critic(env, exp_data, reward_type=args.reward)
        else:
            if args.env_id == "Reacher-v2":
                critic = make_critic(env, exp_data, hid_size=100, reward_type=args.reward, scale=1000)
            if args.env_id == "Walker2d-v2":
                critic = make_critic(env, exp_data, hid_size=30, reward_type=args.reward, scale=100)
            if args.env_id == "HalfCheetah-v2":
                critic = make_critic(env, exp_data, hid_size=30, reward_type=args.reward, scale=1000)
            if args.env_id == "Hopper-v2":
                critic = make_critic(env, exp_data, hid_size=30, reward_type=args.reward, scale=1000)
            if args.env_id == "Ant-v2":
                critic = make_critic(env, exp_data, hid_size=128, reward_type=args.reward, scale=100)
            if args.env_id == "MsPacman-v0":
                critic = make_critic(env, exp_data, hid_size=128, reward_type=args.reward, scale=args.rnd_critic_scale, rnd_cnn_type=args.rnd_cnn_type)

        if args.use_cnn:
            policy = policy_fn_cnn
        else:
            policy = policy_fn
        train(env,
              args.seed,
              policy,
              critic,
              exp_data,
              args.g_step,
              args.d_step,
              args.policy_entcoeff,
              args.num_timesteps,
              save_dir,
              args.pretrained,
              args.BC_max_iter,
              args.gamma,
              rnd_iter,
              dyn_norm,
              task_name,
              args.use_cnn
              )
    elif args.task == 'evaluate':
        runner(env,
               policy_fn,
               args.load_model_path,
               timesteps_per_batch=1024,
               number_trajs=10,
               stochastic_policy=args.stochastic_policy,
               save=args.save_sample
               )
    else:
        raise NotImplementedError
    env.close()


def train(env, seed, policy_fn, reward_giver, dataset,
          g_step, d_step, policy_entcoeff, num_timesteps,
          checkpoint_dir, pretrained, BC_max_iter, gamma, rnd_iter, dyn_norm, task_name=None, use_CNN=False):

    pretrained_weight = None
    if pretrained and (BC_max_iter > 0):
        # Pretrain with behavior cloning
        from baselines.rnd_gail.behavior_clone import learn as bc_learn
        pretrained_weight = bc_learn(env, policy_fn, dataset, task_name, max_iters=BC_max_iter, ckpt_dir=checkpoint_dir)


    from baselines.rnd_gail import trpo_mpi
    # Set up for MPI seed
    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    env.seed(workerseed)
    trpo_mpi.learn(env, policy_fn, reward_giver, dataset, rank,
                   pretrained=pretrained, pretrained_weight=pretrained_weight,
                   g_step=g_step, d_step=d_step,
                   entcoeff=policy_entcoeff,
                   max_timesteps=num_timesteps,
                   ckpt_dir=checkpoint_dir,
                   timesteps_per_batch=2048,
                   max_kl=args.max_kl, cg_iters=10, cg_damping=0.1,
                   gamma=gamma, lam=0.97,
                   vf_iters=5, vf_stepsize=1e-3,
                   task_name=task_name, rnd_iter=rnd_iter, dyn_norm=dyn_norm, mmd=args.reward==2, use_CNN=use_CNN)


def runner(env, policy_func, load_model_path, timesteps_per_batch, number_trajs,
           stochastic_policy, save=False):

    # Setup network
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space)
    U.initialize()
    # Prepare for rollouts
    # ----------------------------------------
    pi.load_policy(load_model_path)

    obs_list = []
    acs_list = []
    len_list = []
    ret_list = []
    for _ in tqdm(range(number_trajs)):
        traj = traj_1_generator(pi, env, timesteps_per_batch, stochastic=stochastic_policy)
        obs, acs, ep_len, ep_ret = traj['ob'], traj['ac'], traj['ep_len'], traj['ep_ret']
        obs_list.append(obs)
        acs_list.append(acs)
        len_list.append(ep_len)
        ret_list.append(ep_ret)
    if stochastic_policy:
        print('stochastic policy:')
    else:
        print('deterministic policy:')
    if save:
        filename = load_model_path.split('/')[-1] + '.' + env.spec.id
        np.savez(filename, obs=np.array(obs_list), acs=np.array(acs_list),
                 lens=np.array(len_list), rets=np.array(ret_list))
    avg_len = sum(len_list)/len(len_list)
    avg_ret = sum(ret_list)/len(ret_list)
    print("Average length:", avg_len)
    print("Average return:", avg_ret)
    print("std:", np.std(ret_list))
    return avg_len, avg_ret


# Sample one trajectory (until trajectory end)
def traj_1_generator(pi, env, horizon, stochastic):

    t = 0
    ac = env.action_space.sample()  # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode

    ob = env.reset()
    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode

    # Initialize history arrays
    obs = []
    rews = []
    news = []
    acs = []

    while True:
        ac, vpred = pi.act(stochastic, ob)
        obs.append(ob)
        news.append(new)
        acs.append(ac)

        ob, rew, new, _ = env.step(ac)
        rews.append(rew)

        cur_ep_ret += rew
        cur_ep_len += 1
        if new or t >= horizon:
            break
        t += 1

    obs = np.array(obs)
    rews = np.array(rews)
    news = np.array(news)
    acs = np.array(acs)
    traj = {"ob": obs, "rew": rews, "new": news, "ac": acs,
            "ep_ret": cur_ep_ret, "ep_len": cur_ep_len}
    return traj


if __name__ == '__main__':
    args = argsparser()
    main(args)
