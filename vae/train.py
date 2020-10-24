from tensorflow.python.ops.gen_batch_ops import Batch
from VAE import VariationalAutoencoder
from utils import *
from sklearn.model_selection import train_test_split

# CIFAR10 samples
# x = read_cifar_data(range(1,6))
# n_samples = len(x)
# x_val = read_cifar_test_data()
# n_val_samples = len(x_val)
# image_dim = x.shape[1:]

# Atari observations
print("Start loading dataset...")
x = read_atari_observations()
x_train, x_test = train_test_split(x, test_size=0.02)
n_samples = len(x_train)
n_test_samples = len(x_test)
image_dim = x.shape[1:]
print("Dataset loaded. Start training...")

def train(conv_layer_specs, dense_layer_specs, latent_dim, learning_rate=0.01,
          batch_size=128, use_shared_weights=False, training_epochs=10, display_step=5):
    """
        Function for training the Autoencoder
        Args:
            inputs:
                hidden_layer_sizes: List containing the sizes of all hidden layers including the input
                                    for example: [784, 500, 500] for MNIST dataset.
                latent_dim: Dimension of the latent space.
                learning_rate: learning rate for the SGD optimization.
                batch_size: batch size used for mini batch training.
                training_epochs: No. of training epochs.
                display_steps: Display for every (display_steps) no. of steps.
    """    
    vae = VariationalAutoencoder(image_dim,
                                 conv_layer_specs,
                                 dense_layer_specs,
                                 latent_dim,
                                 learning_rate,
                                 use_shared_weights=use_shared_weights)
# Train and test data generators
    train_gen = BatchGenerator(x_train, batch_size)
    test_gen = BatchGenerator(x_test, batch_size)
# Training Cycle
    with open("cost_record.csv", "w") as f:
        f.write("Epoch,training_cost,test_cost\n")
    best_val_cost = 1e10
    best_epoch_num = 0
    for epoch in range(training_epochs):
        total_cost = 0.
        total_batches = int(n_samples/batch_size)
    # Loop over all batches
        for i in range(total_batches):
            batch_xs = train_gen.next_batch()
            
        # Fit training using batch data
            cost = vae.fit(batch_xs)
        # Compute total loss
            total_cost += cost / n_samples * batch_size
    
    # Calc validation cost
        val_data = test_gen.get_all_data()
        total_val_cost = vae.evaluate(val_data)

    # Display and log
        with open("cost_record.csv", "a") as f:
            f.write("%4d, %.9f, %.9f\n" % (epoch + 1, total_cost, total_val_cost))
        if (epoch + 1) % display_step == 0:
            print ("Epoch:", '%04d' % (epoch+1),
                   "cost=", "{:.9f}".format(total_cost),
                   "val_cost=", "{:.9f}".format(total_val_cost))
        if total_val_cost < best_val_cost:
            best_val_cost = total_val_cost
            best_epoch_num = epoch + 1
            filename = vae.save_model("model")
    print('Saved model "%s" at epoch %d' % (filename, best_epoch_num))
    return vae

if __name__ == "__main__":
    train([(64, 8, 2), (128, 6, 3), (128, 4, 2), (128, 3, 1)], [1000], 128, learning_rate=0.0001, batch_size=128, use_shared_weights=False ,training_epochs=100, display_step=5)