'''
from baselines/ppo1/mlp_policy.py and add simple modification
(1) add reuse argument
(2) cache the `stochastic` placeholder
'''
import tensorflow as tf
import gym
import numpy as np

import sys
sys.path.append("../../")

from ..common import tf_util as U
from ..common.mpi_running_mean_std import RunningMeanStd
from ..common.distributions import make_pdtype
from ..acktr.utils import dense
from ..common.dataset_plus import normalize, denormalize


class MlpPolicy(object):
    recurrent = False

    def __init__(self, name, *args, **kwargs):
        self.scope = name
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            self._init(*args, **kwargs)

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=False, popart=True):
        # assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[None] + list(ob_space))

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space)

        with tf.variable_scope("popart"):
            self.v_rms = RunningMeanStd(shape=[1])

        obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
        last_out = obz
        # Change first layer of dense to cnn_dense
        last_out = tf.nn.tanh(self.cnn_dense(last_out, hid_size, "vffc1", weight_init=U.normc_initializer(1.0)))
        for i in np.arange(1, num_hid_layers):
            last_out = tf.nn.tanh(dense(last_out, hid_size, "vffc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        self.norm_vpred = dense(last_out, 1, "vffinal", weight_init=U.normc_initializer(1.0))[:, 0]
        if popart:
            self.vpred = denormalize(self.norm_vpred, self.v_rms)
        else:
            self.vpred = self.norm_vpred

        last_out = obz
        # Change first layer of dense to cnn_dense
        last_out = tf.nn.tanh(self.cnn_dense(last_out, hid_size, "polfc1",  weight_init=U.normc_initializer(1.0)))
        for i in np.arange(1, num_hid_layers):
            last_out = tf.nn.tanh(dense(last_out, hid_size, "polfc%i"%(i+1),  weight_init=U.normc_initializer(1.0)))

        if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
            mean = dense(last_out, pdtype.param_shape()[0]//2, "polfinal", U.normc_initializer(0.01))
            logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
            pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
        else:
            pdparam = dense(last_out, pdtype.param_shape()[0], "polfinal", U.normc_initializer(0.01))

        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        # change for BC
        stochastic = U.get_placeholder(name="stochastic", dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self.ac = ac
        self._act = U.function([stochastic, ob], [ac, self.vpred])

        self.use_popart = popart
        if popart:
            self.init_popart()

        ret = tf.placeholder(tf.float32, [None])
        vferr = tf.reduce_mean(tf.square(self.vpred - ret))
        self.vlossandgrad = U.function([ob, ret], U.flatgrad(vferr, self.get_vf_variable()))

    def cnn_dense(self, x, size, name, weight_init=None, bias_init=0, weight_loss_dict=None, reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            assert (len(tf.get_variable_scope().name.split('/')) == 2)

            # # CNN
            # fan_in = [4, 32, 64]
            # fan_out = [32, 64, 64]
            # low, high = [], []
            # for i in range(len(fan_in)):
            #     low.append(-np.sqrt(6.0/(fan_in[i] + fan_out[i])))
            #     high.append(np.sqrt(6.0/(fan_in[i] + fan_out[i])))
            # filters = [
            #             tf.Variable(tf.random_uniform((8, 8, fan_in[0], fan_out[0]), minval=low[0], maxval=high[0], dtype=tf.float32)),
            #             tf.Variable(tf.random_uniform((4, 4, fan_in[1], fan_out[1]), minval=low[1], maxval=high[1], dtype=tf.float32)),
            #             tf.Variable(tf.random_uniform((3, 3, fan_in[2], fan_out[2]), minval=low[2], maxval=high[2], dtype=tf.float32))
            # ]
            # strides = [[1,4,4,1], [1,2,2,1], [1,1,1,1]]
            filters, strides = U.cnn()
            
            cnn_layer = tf.nn.conv2d(x, filters[0], strides=strides[0], padding="VALID")
            assert len(filters) > 1 and len(strides) == len(filters)
            for i in np.arange(1, len(filters)):
                cnn_layer = tf.nn.conv2d(cnn_layer, filters[i], strides[i], "VALID")
            x = tf.reshape(cnn_layer, [-1, int(np.prod(cnn_layer.shape[1:]))])   # flatten cnn output, except the batch axis
            # CNN ends

            w = tf.get_variable("w", [x.get_shape()[1], size], initializer=weight_init)
            b = tf.get_variable("b", [size], initializer=tf.constant_initializer(bias_init))
            weight_decay_fc = 3e-4

            if weight_loss_dict is not None:
                weight_decay = tf.multiply(tf.nn.l2_loss(w), weight_decay_fc, name='weight_decay_loss')
                if weight_loss_dict is not None:
                    weight_loss_dict[w] = weight_decay_fc
                    weight_loss_dict[b] = 0.0

                tf.add_to_collection(tf.get_variable_scope().name.split('/')[0] + '_' + 'losses', weight_decay)

            return tf.nn.bias_add(tf.matmul(x, w), b)

    def init_popart(self):
        old_std = tf.placeholder(tf.float32, shape=[1], name='old_std')
        new_std = self.v_rms.std
        old_mean = tf.placeholder(tf.float32, shape=[1], name='old_mean')
        new_mean = self.v_rms.mean

        renormalize_Q_outputs_op = []
        vs = self.output_vars
        M, b = vs
        renormalize_Q_outputs_op += [M.assign(M * old_std / new_std)]
        renormalize_Q_outputs_op += [b.assign((b * old_std + old_mean - new_mean) / new_std)]
        self.renorm_v = U.function([old_std, old_mean], [], updates=renormalize_Q_outputs_op)

    def act(self, stochastic, ob):
        ac1, vpred1 = self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.trainable_variables(self.scope)

    def get_initial_state(self):
        return []

    def get_vf_variable(self):
        return tf.trainable_variables(self.scope + "/vf")

    def update_popart(self, v_targets):
        old_mean, old_std = U.get_session().run([self.v_rms.mean, self.v_rms.std])
        self.v_rms.update(v_targets)
        self.renorm_v(old_std, old_mean)

    @property
    def output_vars(self):
        output_vars = [var for var in self.get_vf_variable() if 'vffinal' in var.name]
        return output_vars

    def save_policy(self, name):
        U.save_variables(name, variables=self.get_variables())

    def load_policy(self, name):
        U.load_variables(name, variables=self.get_variables())
