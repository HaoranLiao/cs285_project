'''
from baselines/ppo1/mlp_policy.py and add simple modification
(1) add reuse argument
(2) cache the `stochastic` placeholder
'''
import tensorflow as tf
import gym

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

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=False, popart=True, activation="tanh"):
        assert isinstance(ob_space, gym.spaces.Box)

        if activation == "tanh":
            ac_fn = tf.nn.tanh
        elif activation == "relu":
            ac_fn = tf.nn.relu
        if num_hid_layers > 1 and type(hid_size) is int:
            hid_size = [hid_size] * num_hid_layers

        self.pdtype = pdtype = make_pdtype(ac_space)

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[None] + list(ob_space.shape))

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        with tf.variable_scope("popart"):
            self.v_rms = RunningMeanStd(shape=[1])

        obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
        last_out = obz
        for i in range(num_hid_layers):
            last_out = ac_fn(dense(last_out, hid_size[i], "vffc%i" % (i+1), weight_init=U.normc_initializer(1.0)))
        self.norm_vpred = dense(last_out, 1, "vffinal", weight_init=U.normc_initializer(1.0))[:, 0]
        if popart:
            self.vpred = denormalize(self.norm_vpred, self.v_rms)
        else:
            self.vpred = self.norm_vpred

        last_out = obz
        for i in range(num_hid_layers):
            last_out = ac_fn(dense(last_out, hid_size[i], "polfc%i" % (i+1), weight_init=U.normc_initializer(1.0)))

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
