
from keras.datasets import boston_housing
from keras import models, layers, optimizers
import matplotlib.pyplot as plt
import numpy as np
#training set
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print(x_train.shape)
x_train[0]

# Hyperparameters
h_num_units = 13 #13

# More info about initializers: https://keras.io/initializers/
h_kernel_initializer = 'normal'

# More info about activation functions: https://keras.io/activations/
h_activation = 'tanh' #sigmoid, tanh, softmax

# More info about optimizers: https://keras.io/optimizers/
h_learning_rate = 0.1 #0.1, 0.0001

h_optimizer = optimizers.Adam(lr=h_learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0000, amsgrad=False)
#h_optimizer = optimizers.SGD(lr=h_learning_rate, momentum=0.0, decay=0.0, nesterov=False)

h_epochs = 10 #10
h_batchsize = 32 #32

model = models.Sequential()

# First hidden layer
model.add(layers.Dense(
    h_num_units, 
    input_dim=13, 
    kernel_initializer=h_kernel_initializer, 
    activation=h_activation
))

# Second hidden layer
# h2_num_units = 30
# model.add(layers.Dense(
#     h2_num_units, 
#     kernel_initializer=h_kernel_initializer, 
#     activation=h_activation
# ))

# Output layer
model.add(layers.Dense(1, kernel_initializer='normal'))

model.compile(loss='mean_squared_error', optimizer=h_optimizer, metrics=['mse'])
print(model.summary())

#%%time
history = model.fit(x_train, y_train, validation_split=(0.2), epochs=h_epochs, batch_size=h_batchsize, verbose=2)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Boston House Prices Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

model.evaluate(x_test, y_test, batch_size=h_batchsize, verbose=2)