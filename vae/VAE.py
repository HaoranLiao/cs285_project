import numpy as np
import tensorflow as tf

class VariationalAutoencoder:
    """
        Variational Autoencoder (VAE)
        Args:
            inputs:
                hidden_layer_sizes: List containing the sizes of all hidden layers including the input
                                    for example: [784, 500, 500] for MNIST dataset.
                latent_dim: Dimension of the latent space.
                learning_rate: learning rate for the SGD optimization.
                batch_size: batch size used for mini batch training.
            outputs:
                cost: loss obtained for the respective batch size.
    """
    def __init__(self, image_dim, conv_layer_specs, dense_layer_specs ,latent_dim, learning_rate=0.001, use_shared_weights=False):
        self.image_dim = image_dim
        self.conv_layers = conv_layer_specs
        self.dense_layer_specs = dense_layer_specs
        self.conved_shapes = self.calc_conv_shapes(image_dim, conv_layer_specs)
        self.flattened_shape = np.prod(self.conved_shapes[-1])
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.use_shared_weights = use_shared_weights
    # Tensor flow graph input
        self.x = tf.placeholder(tf.float32, (None,) + image_dim)
    # The autoencoder model
        self.create_model()
    # Optimizer and loss function
        self.loss_optimizer()
    # Initialize the tensorflow computaional graph
        init = tf.global_variables_initializer()
    # Launch a session
        self.sess = tf.InteractiveSession()
        self.sess.run(init)
        
    def create_model(self):
    # Initialize the parameters of encoder
        enc_weights, enc_biases = self.initialize_enc_weights()
        self.enc_weights = enc_weights
        self.enc_biases = enc_biases
    # Encoder
        self.z_mean, self.z_log_sigma = self.encoder(self.x, enc_weights, enc_biases)
    # Sampling
        self.z = self.sampling(self.z_mean, self.z_log_sigma)
    # Initialize the parameters of decoder
        dec_weights, dec_biases = self.initialize_dec_weights(use_shared_weights=self.use_shared_weights)
        self.dec_weights = dec_weights
        self.dec_biases = dec_biases
    # Decoder
        self.recon = self.decoder(self.z, dec_weights, dec_biases)
    

    def save_model(self, save_addr):
        saver = tf.train.Saver()
        saved_filename = saver.save(self.sess, save_addr)
        return saved_filename

    def load_model(self, load_addr):
        saver = tf.train.Saver()
        saver.restore(self.sess, load_addr)
        
    def calc_conv_shapes(self, input_dim, conv_specs):
        width, height, channel = input_dim
        conved_shapes = []
        for unit, filter_size, stride in conv_specs:
            width = int((width - filter_size) / stride) + 1
            height = int((height - filter_size) / stride) + 1
            channel = unit
            conved_shapes.append((width, height, channel))
        return conved_shapes
    
    def initialize_enc_weights(self):
    # Encoder Weights
        weights = {}
        biases = {}
    # Conv layer weights and biases
        for i in range(len(self.conv_layers)):
            conv_spec = self.conv_layers[i]
            if i != 0:
                input_channels = self.conv_layers[i - 1][0]
            else:
                input_channels = self.image_dim[-1]
            output_channels = conv_spec[0]
            filter_size = conv_spec[1]
            low = -np.sqrt(6.0/(input_channels + output_channels))
            high = np.sqrt(6.0/(input_channels + output_channels))
            weights[i] = tf.Variable(tf.random_uniform((filter_size, filter_size, input_channels, output_channels),
                                                       minval=low,
                                                       maxval=high,
                                                       dtype=tf.float32)
                                    )
            biases[i] = tf.Variable(tf.zeros([output_channels]), dtype=tf.float32)
        
    # Dense layer weights and biases
        for i in range(len(self.dense_layer_specs)):
            fan_in = self.dense_layer_specs[i - 1] if i != 0 else self.flattened_shape
            fan_out = self.dense_layer_specs[i]
            low = -np.sqrt(6.0/(fan_in + fan_out))
            high = np.sqrt(6.0/(fan_in + fan_out))
            weights[f'fc_{i}'] = tf.Variable(tf.random_uniform((fan_in, fan_out),
                                                      minval=low,
                                                      maxval=high,
                                                      dtype=tf.float32)
                                   )
            biases[f'fc_{i}'] = tf.Variable(tf.zeros([fan_out]), dtype=tf.float32)
    # mu and sigma layer weights
        if len(self.dense_layer_specs) > 0:
            fan_in = self.dense_layer_specs[-1]
        else:
            fan_in = self.flattened_shape
        fan_out = self.latent_dim
        low = -np.sqrt(6.0/(fan_in + fan_out))
        high = np.sqrt(6.0/(fan_in + fan_out))
        weights['mu'] = tf.Variable(tf.random_uniform((fan_in, fan_out),
                                                      minval=low,
                                                      maxval=high,
                                                      dtype=tf.float32)
                                   )
        biases['mu'] = tf.Variable(tf.zeros([fan_out]), dtype=tf.float32)
        weights['sigma'] = tf.Variable(tf.random_uniform((fan_in, fan_out),
                                                         minval=low,
                                                         maxval=high,
                                                         dtype=tf.float32)
                                      )
        biases['sigma'] = tf.Variable(tf.zeros([fan_out]), dtype=tf.float32)
        
        return weights, biases
        
    
    
    def initialize_dec_weights(self, use_shared_weights):
    # Decoder Weights
        weights = {}
        biases = {}
    # mu and sigma layer weights
        fan_in = self.latent_dim
        if len(self.dense_layer_specs) > 0:
            fan_out = self.dense_layer_specs[-1]
        else:
            fan_out = self.flattened_shape
        low = -np.sqrt(6.0/(fan_in + fan_out))
        high = np.sqrt(6.0/(fan_in + fan_out))
        weights['z'] = tf.Variable(tf.random_uniform((fan_in, fan_out),
                                                     minval=low,
                                                     maxval=high,
                                                     dtype=tf.float32)
                                  )
        biases['z'] = tf.Variable(tf.zeros([fan_out]), dtype=tf.float32)
    # Dense layers
        for i in range(len(self.dense_layer_specs) - 1, -1, -1):
            fan_in = self.dense_layer_specs[i]
            fan_out = self.dense_layer_specs[i - 1] if i != 0 else self.flattened_shape
            low = -np.sqrt(6.0/(fan_in + fan_out))
            high = np.sqrt(6.0/(fan_in + fan_out))
            weights[f'fc_{i}'] = tf.Variable(tf.random_uniform((fan_in, fan_out),
                                                     minval=low,
                                                     maxval=high,
                                                     dtype=tf.float32)
                                  )
            biases[f'fc_{i}'] = tf.Variable(tf.zeros([fan_out]), dtype=tf.float32)
    
    # Conv layer weights and biases
        if use_shared_weights:
            for i in range(len(self.conv_layers) - 1, -1, -1):
                weights[i] = self.enc_weights[i]
                if i != 0:
                    fan_out = self.conv_layers[i - 1][0]
                else:
                    fan_out = self.image_dim[-1]
                biases[i] = tf.Variable(tf.zeros([fan_out]), dtype=tf.float32)
        else:
            for i in range(len(self.conv_layers) - 1, 0, -1):
                filter_size = self.conv_layers[i][1]
                fan_in = self.conv_layers[i][0]
                fan_out = self.conv_layers[i - 1][0]
                low = -np.sqrt(6.0/(fan_in + fan_out))
                high = np.sqrt(6.0/(fan_in + fan_out))
                weights[i] = tf.Variable(tf.random_uniform((filter_size, filter_size, fan_out, fan_in),
                                                        minval=low,
                                                        maxval=high,
                                                        dtype=tf.float32)
                                        )
                biases[i] = tf.Variable(tf.zeros([fan_out]), dtype=tf.float32)
            filter_size = self.conv_layers[0][1]
            fan_in = self.conv_layers[0][0]
            fan_out = self.image_dim[-1]
            low = -np.sqrt(6.0/(fan_in + fan_out))
            high = np.sqrt(6.0/(fan_in + fan_out))
            weights[0] = tf.Variable(tf.random_uniform((filter_size, filter_size, fan_out, fan_in),
                                                        minval=low,
                                                        maxval=high,
                                                        dtype=tf.float32)
                                    )
            biases[0] = tf.Variable(tf.zeros([fan_out]), dtype=tf.float32)
        return weights, biases
        
    def sampling(self, z_mean, z_log_sigma):
        epsilon = tf.random_normal((tf.shape(z_mean)[0], self.latent_dim), 
                                   0, 1, dtype=tf.float32)
        z = tf.add(z_mean, tf.multiply(tf.exp(z_log_sigma), epsilon))
        return z

    def encoder(self, x, weights, biases):
        hidden = x
        for i in range(len(self.conv_layers)):
            hidden = tf.nn.conv2d(hidden, weights[i], strides=self.conv_layers[i][-1], padding="VALID") + biases[i]
            hidden = tf.nn.relu(hidden)
        hidden = tf.reshape(hidden, (-1, self.flattened_shape))
        for i in range(len(self.dense_layer_specs)):
            hidden = tf.add(tf.matmul(hidden, weights[f'fc_{i}']), biases[f'fc_{i}'])
            hidden = tf.nn.relu(hidden)
        z_mean = tf.add(tf.matmul(hidden, weights['mu']), biases['mu'])
        z_log_sigma = tf.add(tf.matmul(hidden, weights['sigma']), biases['sigma'])
        return z_mean, z_log_sigma
    
    # def encoder(self, x, weights, biases):
    #     h = x
    #     for i in range(len(self.hidden_layer_sizes)-1):
    #         h = tf.add(tf.matmul(h, weights[i]), biases[i])
    #         h = tf.nn.relu(h)
    #     z_mean = tf.add(tf.matmul(h, weights['mu']), biases['mu'])
    #     z_log_sigma = tf.add(tf.matmul(h, weights['sigma']), biases['sigma'])
        
    #     return z_mean, z_log_sigma

    def decoder(self, z, weights, biases):
        hidden = tf.add(tf.matmul(z, weights['z']), biases['z'])
        hidden = tf.nn.relu(hidden)
        for i in range(len(self.dense_layer_specs) - 1, -1, -1):
            hidden = tf.add(tf.matmul(hidden, weights[f'fc_{i}']), biases[f'fc_{i}'])
            hidden = tf.nn.relu(hidden)
        hidden = tf.reshape(hidden, (-1, ) + self.conved_shapes[-1])
        for i in range(len(self.conv_layers) - 1, 0, -1):
            hidden = tf.nn.conv2d_transpose(hidden, weights[i], output_shape=(tf.shape(z)[0],) + self.conved_shapes[i - 1], strides=self.conv_layers[i][-1], padding="VALID") + biases[i]
            hidden = tf.nn.relu(hidden)
        output = tf.nn.conv2d_transpose(hidden, weights[0], output_shape=(tf.shape(z)[0],) + self.image_dim, strides=self.conv_layers[0][-1], padding="VALID") + biases[0]
        output = tf.nn.sigmoid(output)
        return output
    
    # def decoder(self, z, weights, biases):
    #     h = tf.add(tf.matmul(z, weights['z']), biases['z'])
    #     for i in range(len(self.hidden_layer_sizes)-1)[::-1]:
    #         h = tf.add(tf.matmul(h, weights[i]), biases[i])
    #         h = tf.nn.relu(h)
    #     return h
    
    def loss_optimizer(self):
        x_flatten = tf.reshape(self.x, (tf.shape(self.x)[0], -1))
        reconstructed_x_flatten = tf.reshape(self.recon, (tf.shape(self.recon)[0], -1))
        recon_loss = tf.reduce_sum(tf.square(x_flatten - reconstructed_x_flatten), axis=1)
        latent_loss = -0.5*tf.reduce_sum(1 + self.z_log_sigma
                                         - tf.square(self.z_mean)
                                         - tf.exp(self.z_log_sigma), axis=1)
        self.cost = tf.reduce_mean(recon_loss + latent_loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def fit(self, X):
        _, cost = self.sess.run([self.optimizer, self.cost], feed_dict = {self.x: X})
        return cost

    def reconstruct(self, input_image):
        reconstructed_image = self.sess.run(self.recon, feed_dict={self.x: input_image})
        return reconstructed_image

    def evaluate(self, X):
        cost = self.sess.run(self.cost, feed_dict = {self.x: X})
        return cost

    def encode(self, X):
        mean = self.sess.run(self.z_mean, feed_dict = {self.x: X})
        return mean