import tensorflow as tf
import sys
sys.path.append("../../")
from baselines.common import tf_util as U
from baselines.common.dataset import iterbatches
from baselines import logger
import baselines.common.tf_util as U

import numpy as np
from tqdm import tqdm

class RND_Critic_CNN(object):
    def __init__(self, ob_size, ac_size, rnd_hid_size=128, rnd_hid_layer=4, hid_size=64, hid_layer=1,
                 out_size=64, scale=250000.0, offset=0., reward_scale=1.0, scope="rnd", rnd_cnn_type=1):
        self.ob_size = ob_size
        self.ac_size = ac_size
        self.scope = scope
        self.scale = scale
        self.offset = offset
        self.out_size = out_size
        self.rnd_hid_size = rnd_hid_size
        self.rnd_hid_layer = rnd_hid_layer
        self.hid_size = hid_size
        self.hid_layer = hid_layer
        self.reward_scale = reward_scale
        self.rnd_cnn_type = rnd_cnn_type
        print("RND Critic")

        ob = tf.placeholder(tf.float32, (None,) + ob_size)
        ac = tf.placeholder(tf.float32, (None, ac_size))
        lr = tf.placeholder(tf.float32, None)

        feat = self.build_graph(ob, ac, self.scope, hid_layer, hid_size, out_size)
        rnd_feat = self.build_graph(ob, ac, self.scope+"_rnd", rnd_hid_layer, rnd_hid_size, out_size)

        feat_loss = tf.reduce_mean(tf.square(feat-rnd_feat))
        self.reward = reward_scale*tf.exp(offset- tf.reduce_mean(tf.square(feat - rnd_feat), axis=-1) * self.scale)

        rnd_loss = tf.reduce_mean(tf.square(feat - rnd_feat), axis=-1) * self.scale
        # self.reward = reward_scale * tf.exp(offset - rnd_loss)
        # self.reward = reward_scale * (tf.math.softplus(rnd_loss) - rnd_loss)
        self.reward_func = U.function([ob, ac], self.reward)
        self.raw_reward = U.function([ob, ac], rnd_loss)

        self.trainer = tf.train.AdamOptimizer(learning_rate=lr)

        gvs = self.trainer.compute_gradients(feat_loss, self.get_trainable_variables())

        self._train = U.function([ob, ac, lr], [], updates=[self.trainer.apply_gradients(gvs)])

        feat_loss = tf.reduce_mean(tf.square(feat - rnd_feat))
        feat_loss_no_reduce = tf.reduce_mean(tf.square(feat - rnd_feat), axis=-1)
        self.feat_loss_fn = U.function([ob, ac], feat_loss)
        self.feat_loss_no_reduce_fn = U.function([ob, ac], feat_loss_no_reduce)

        feat = tf.reduce_mean(feat)
        self.feat_fn = U.function([ob, ac], feat)
        rnd_feat = tf.reduce_mean(rnd_feat)
        self.rnd_feat_fn = U.function([ob, ac], rnd_feat)

    def build_graph(self, ob, ac, scope, hid_layer, hid_size, out_size):
        filters, strides, cnn_type = U.cnn(self.rnd_cnn_type)
        logger.log(f'critic cnn type: {cnn_type}')
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            cnn_layer = tf.nn.conv2d(ob, filters[0], strides=strides[0], padding="VALID")
            assert len(filters) > 1 and len(strides) == len(filters)
            for i in np.arange(1, len(filters)):
                cnn_layer = tf.nn.conv2d(cnn_layer, filters[i], strides[i], "VALID")
            ob = tf.reshape(cnn_layer, [-1, int(np.prod(cnn_layer.shape[1:]))])   # flatten cnn output, except the batch axis #1100+
            logger.log(f"critic cnn ob output shape: {ob.shape}")

            layer = ob
            list_of_output_shape = [500, 100]  # 1000 -> 500 -> 100
            logger.log(f"critic cnn dense: {list_of_output_shape}")
            weights, biases = U.dense(layer, list_of_output_shape)
            for i in range(len(list_of_output_shape) - 1):
                layer = tf.add(tf.matmul(layer, weights[i]), biases[i])
                layer = tf.nn.relu(layer)
            layer = tf.add(tf.matmul(layer, weights[-1]), biases[-1])
            ob = layer

            layer = tf.concat([ob, ac], axis=1)
            for _ in range(hid_layer):
                layer = tf.layers.dense(layer, hid_size, activation=tf.nn.leaky_relu)
            layer = tf.layers.dense(layer, out_size, activation=None)
            logger.log(f"[ob, ac] dense hid_layer: {hid_layer}, hid_size: {hid_size}, out_size: {out_size}")
        return layer

    def build_reward_op(self, ob, ac):
        feat = self.build_graph(ob, ac, self.scope, self.hid_layer, self.hid_size, self.out_size)
        rnd_feat = self.build_graph(ob, ac, self.scope + "_rnd", self.rnd_hid_layer, self.rnd_hid_size
                                    , self.out_size)

        reward = self.reward_scale* tf.exp(self.offset- tf.reduce_mean(tf.square(feat - rnd_feat), axis=-1) * self.scale)
        return reward

    def get_trainable_variables(self):
        return tf.trainable_variables(self.scope)

    def get_reward(self, ob, ac):
        return self.reward_func(ob, ac)
    
    def get_raw_reward(self, ob, ac):
        return self.raw_reward(ob, ac)

    def get_feature_loss(self, ob, ac, reduce=True):
        if reduce:
            return self.feat_loss_fn(ob, ac)
        else:
            return self.feat_loss_no_reduce_fn(ob, ac)

    def get_feat(self, ob, ac):
        return self.feat_fn(ob, ac)

    def get_rnd_feat(self, ob, ac):
        return self.rnd_feat_fn(ob, ac)

    def train(self, ob, ac, batch_size=32, lr=0.0001, iter=200):
        logger.info("Training RND Critic")
        indices = np.arange(len(ob))
        np.random.shuffle(indices)
        inspection_set = [ob[indices[:1000]], ac[indices[:1000]]]
        out_of_dist_set = [ob[indices[:1000]], np.random.random(size=(inspection_set[1].shape))]
        logger.info("iter, in_dist_loss, out_of_dist_loss,\t in_feat, in_rnd_feat,\t out_feat, out_rnd_feat")
        in_dist_loss = self.get_feature_loss(*inspection_set)
        out_of_dist_loss = self.get_feature_loss(*out_of_dist_set)
        in_feat = self.get_feat(*inspection_set)
        in_rnd_feat = self.get_rnd_feat(*inspection_set)
        out_feat = self.get_feat(*out_of_dist_set)
        out_rnd_feat = self.get_rnd_feat(*out_of_dist_set)
        logger.info("%d,%f,%f,\t%f,%f,\t%f, %f"%(0,in_dist_loss,out_of_dist_loss, in_feat, in_rnd_feat, out_feat, out_rnd_feat))
        for i in tqdm(range(iter)):
            for data in iterbatches([ob, ac], batch_size=batch_size, include_final_partial_batch=True):
                self._train(*data, lr)
            in_dist_loss = self.get_feature_loss(*inspection_set)
            out_of_dist_loss = self.get_feature_loss(*out_of_dist_set)
            in_feat = self.get_feat(*inspection_set)
            in_rnd_feat = self.get_rnd_feat(*inspection_set)
            out_feat = self.get_feat(*out_of_dist_set)
            out_rnd_feat = self.get_rnd_feat(*out_of_dist_set)
            logger.info("%d,%f,%f,\t%f,%f,\t%f, %f"%(i+1 ,in_dist_loss,out_of_dist_loss, in_feat, in_rnd_feat, out_feat, out_rnd_feat))

    def save_trained_variables(self, save_addr):
        saver = tf.train.Saver(self.get_trainable_variables())
        saver.save(U.get_session(), save_addr)

    def load_trained_variables(self, load_addr):
        saver = tf.train.Saver(self.get_trainable_variables())
        saver.restore(U.get_session(), load_addr)

class RND_Critic(object):
    def __init__(self, ob_size, ac_size, rnd_hid_size=128, rnd_hid_layer=4, hid_size=64, hid_layer=1,
                 out_size=64, scale=250000.0, offset=0., reward_scale=1.0, scope="rnd"):
        self.scope = scope
        self.scale = scale
        logger.info("scale: %f" % scale)
        self.offset = offset
        self.out_size = out_size
        self.rnd_hid_size = rnd_hid_size
        self.rnd_hid_layer = rnd_hid_layer
        self.hid_size = hid_size
        self.hid_layer = hid_layer
        self.reward_scale = reward_scale
        print("RND Critic")

        ob = tf.placeholder(tf.float32, [None, ob_size])
        ac = tf.placeholder(tf.float32, [None, ac_size])
        lr = tf.placeholder(tf.float32, None)


        feat = self.build_graph(ob, ac, self.scope, hid_layer, hid_size, out_size)
        rnd_feat = self.build_graph(ob, ac, self.scope+"_rnd", rnd_hid_layer, rnd_hid_size, out_size)

        feat_loss = tf.reduce_mean(tf.square(feat-rnd_feat))
        feat_loss_no_reduce = tf.reduce_mean(tf.square(feat-rnd_feat), axis=-1)
        self.reward = reward_scale*tf.exp(offset- tf.reduce_mean(tf.square(feat - rnd_feat), axis=-1) * self.scale)

        rnd_loss = tf.reduce_mean(tf.square(feat - rnd_feat), axis=-1) * self.scale
        # self.reward = reward_scale * tf.exp(offset - rnd_loss)
        # self.reward = reward_scale * (tf.math.softplus(rnd_loss) - rnd_loss)
        self.reward_func = U.function([ob, ac], self.reward)
        self.raw_reward = U.function([ob, ac], rnd_loss)
        self.feat_loss_fn = U.function([ob, ac], feat_loss)
        self.feat_loss_no_reduce_fn = U.function([ob, ac], feat_loss_no_reduce)

        self.trainer = tf.train.AdamOptimizer(learning_rate=lr)

        gvs = self.trainer.compute_gradients(feat_loss, self.get_trainable_variables())

        self._train = U.function([ob, ac, lr], [], updates=[self.trainer.apply_gradients(gvs)])

    def build_graph(self, ob, ac, scope, hid_layer, hid_size, size):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            layer = tf.concat([ob, ac], axis=1)
            for _ in range(hid_layer):
                layer = tf.layers.dense(layer, hid_size, activation=tf.nn.leaky_relu)
            layer = tf.layers.dense(layer, size, activation=None)
        return layer

    def build_reward_op(self, ob, ac):
        feat = self.build_graph(ob, ac, self.scope, self.hid_layer, self.hid_size, self.out_size)
        rnd_feat = self.build_graph(ob, ac, self.scope + "_rnd", self.rnd_hid_layer, self.rnd_hid_size
                                    , self.out_size)

        reward = self.reward_scale* tf.exp(self.offset- tf.reduce_mean(tf.square(feat - rnd_feat), axis=-1) * self.scale)
        return reward

    def get_trainable_variables(self):
        return tf.trainable_variables(self.scope)


    def get_reward(self, ob, ac):
        return self.reward_func(ob, ac)

    def get_feature_loss(self, ob, ac, reduce=True):
        if reduce:
            return self.feat_loss_fn(ob, ac)
        else:
            return self.feat_loss_no_reduce_fn(ob, ac)
    
    
    def get_raw_reward(self, ob, ac):
        return self.raw_reward(ob, ac)

    def train(self, ob, ac, batch_size=32, lr=0.0001, iter=200):
        logger.info("Training RND Critic")
        # indices = np.arange(len(ob))
        # np.random.shuffle(indices)
        # inspection_set = [ob[indices[:1000]], ac[indices[:1000]]]
        # out_of_dist_set = [ob[indices[:1000]], np.random.random(size=(inspection_set[1].shape))]
        # logger.info("iter, in_dist_loss, out_of_dist_loss")
        # in_dist_loss = self.get_feature_loss(*inspection_set)
        # out_of_dist_loss = self.get_feature_loss(*out_of_dist_set)
        # logger.info("%d,%f,%f"%(0,in_dist_loss,out_of_dist_loss))
        for i in tqdm(range(iter)):
        # for i in range(iter):
            for data in iterbatches([ob, ac], batch_size=batch_size, include_final_partial_batch=True):
                self._train(*data, lr)
            # in_dist_loss = self.get_feature_loss(*inspection_set)
            # out_of_dist_loss = self.get_feature_loss(*out_of_dist_set)
            # logger.info("%d,%f,%f"%(i+1,in_dist_loss,out_of_dist_loss))


    def save_trained_variables(self, save_addr):
        saver = tf.train.Saver(self.get_trainable_variables())
        saver.save(U.get_session(), save_addr)

    def load_trained_variables(self, load_addr):
        saver = tf.train.Saver(self.get_trainable_variables())
        saver.restore(U.get_session(), load_addr)

class Enc_Critic(object):
    def __init__(self, ob_size, ac_size, hid_size=128, hid_layer=1, scale=250000.0, offset=0., reward_scale=1.0,
                 reg_scale=0.0001, scope="enc"):
        self.scope = scope
        self.scale = scale
        self.offset = offset
        self.out_size = ob_size+ac_size
        self.hid_size = hid_size
        self.hid_layer = hid_layer
        self.reward_scale = reward_scale
        print("Enc Critic")

        ob = tf.placeholder(tf.float32, [None, ob_size])
        ac = tf.placeholder(tf.float32, [None, ac_size])
        lr = tf.placeholder(tf.float32, None)

        target = tf.concat([ob, ac], axis=1)
        feat = self.build_graph(ob, ac, self.scope, hid_layer, hid_size, self.out_size)


        feat_loss = tf.reduce_mean(tf.square(feat - target))
        self.reward = reward_scale * tf.exp(offset - tf.reduce_mean(tf.square(feat - target), axis=-1) * self.scale)

        raw_loss = tf.reduce_mean(tf.square(feat - target), axis=-1) * self.scale
        self.reward_func = U.function([ob, ac], self.reward)
        self.raw_reward = U.function([ob, ac], raw_loss)

        self.trainer = tf.train.AdamOptimizer(learning_rate=lr)
        if reg_scale>0:
            feat_loss +=tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(reg_scale),
                                                   weights_list=self.get_trainable_variables())

        gvs = self.trainer.compute_gradients(feat_loss, self.get_trainable_variables())

        self._train = U.function([ob, ac, lr], [], updates=[self.trainer.apply_gradients(gvs)])

    def build_graph(self, ob, ac, scope, hid_layer, hid_size, size):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            layer = tf.concat([ob, ac], axis=1)
            for _ in range(hid_layer):
                layer = tf.layers.dense(layer, hid_size, activation=tf.nn.leaky_relu)
            layer = tf.layers.dense(layer, size, activation=None)
        return layer

    def build_reward_op(self, ob, ac):
        feat = self.build_graph(ob, ac, self.scope, self.hid_layer, self.hid_size, self.out_size)
        target = tf.concat([ob, ac], axis=1)
        reward = self.reward_scale * tf.exp(
            self.offset - tf.reduce_mean(tf.square(feat - target), axis=-1) * self.scale)
        return reward

    def get_trainable_variables(self):
        return tf.trainable_variables(self.scope)

    def get_reward(self, ob, ac):
        return self.reward_func(ob, ac)

    def get_raw_reward(self, ob, ac):
        return self.raw_reward(ob, ac)

    def train(self, ob, ac, batch_size=32, lr=0.001, iter=200):
        logger.info("Training RND Critic")
        for _ in range(iter):
            for data in iterbatches([ob, ac], batch_size=batch_size, include_final_partial_batch=True):
                self._train(*data, lr)

