from keras.datasets.fashion_mnist import load_data
(trainX, trainY), (testX, testY) = load_data()

print("Train:", trainX.shape, trainY.shape)
print("Test:", testX.shape, testY.shape)
# %%
# plot a few example images
import matplotlib.pyplot as plt

for i in range(25):
  plt.subplot(5,5,1+i)
  plt.axis('off')
  plt.imshow(trainX[i], cmap='gray_r')

plt.show()
# %%
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint

def load_real_samples():
  # load dataset
  (trainX, _), (_, _) = load_data()
  X = expand_dims(trainX, axis=-1)
  X = X.astype('float32')
  # scale from [0,255] to [-1,1]
  X = (X-127.5)/127.5
  return X

def generate_real_samples(dataset, n_samples):
  # choose random instances
  ix = randint(0, dataset.shape[0], n_samples)
  # select images
  X = dataset[ix]
  # generate class labels
  y = ones((n_samples,1))
  return X, y

def generate_latent_points(latent_dim, n_samples):
  x_input = randn(latent_dim * n_samples)
  # reshape
  x_input = x_input.reshape(n_samples, latent_dim)
  return x_input

def generate_fake_samples(generator, latent_dim, n_samples):
  # generate points in latent space
  x_input = generate_latent_points(latent_dim, n_samples)
  # predict outputs
  X = generator.predict(x_input)
  # create class labels
  y = zeros((n_samples,1))
  return X, y
# %%
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout

def define_disciminator(in_shape=(28,28,1)):
  model = Sequential()
  # downsampling
  model.add(Conv2D(128, (3,3), strides=(2,2), padding='same', input_shape=in_shape))
  model.add(LeakyReLU(alpha=0.2))
  # classify
  model.add(Flatten())
  model.add(Dropout(0.4))
  model.add(Dense(1, activation='sigmoid'))
  # compile model
  opt = Adam(lr=0.0002, beta_1=0.5)
  model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
  return model

def define_generator(latent_dim):
  model = Sequential()
  n_nodes = 128*7*7
  model.add(Dense(n_nodes, input_dim=latent_dim))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Reshape((7,7,128)))
  # upsample the 7x7 maps into 14x14
  model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
  model.add(LeakyReLU(alpha=0.2))
  # upsample the 14x14 maps into 28x28
  model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
  model.add(LeakyReLU(alpha=0.2))
  # generate
  model.add(Conv2D(1, (7,7), activation='tanh', padding='same'))
  return model

# this composite model is used to train the generator, using the output and error calculated from discriminator
def define_gan(generator, discriminator):
  # make discriminator weights untrainable
  discriminator.trainable = False

  model = Sequential()
  model.add(generator)
  model.add(discriminator)
  # compile model
  opt = Adam(lr=0.0002, beta_1=0.5)
  model.compile(loss='binary_crossentropy', optimizer=opt)
  return model

# %%
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=128):
  bat_per_epo = int(dataset.shape[0]/n_batch)
  half_batch = int(n_batch/2)
  
  for i in range(n_epochs):
    for j in range(bat_per_epo):
      X_real, y_real = generate_real_samples(dataset, half_batch)
      # update discriminator model weights
      d_loss1, _ = d_model.train_on_batch(X_real, y_real)
      # generate 'fake examples'
      X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
      # train discriminator weights on fake samples
      d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
      # prepare points in latent space as input for generator
      X_gan = generate_latent_points(latent_dim, n_batch)
      # inverted labels
      y_gan = ones((n_batch, 1))
      # update generator via discriminator's error
      g_loss = gan_model.train_on_batch(X_gan, y_gan)
      print('>Epoch: %d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' % (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
  g_model.save('generator.h5')
  
# %%
latent_dim = 100
discriminator = define_disciminator()
generator = define_generator(latent_dim)
gan_model = define_gan(generator, discriminator)

dataset = load_real_samples()
# %%
train(generator, discriminator, gan_model, dataset, latent_dim)