import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

## generate time series dataset
def generate_time_series(batch_size, n_steps):
    # uniform distribution between 0 and 1
    freq1, freq2, offset1, offset2= np.random.rand(4, batch_size, 1)  
    time= np.linspace(0, 1, n_steps)
    series= 0.5*np.sin((time-offset1)*(freq1*10 + 10)) # sine curve 1
    series += 0.2*np.sin((time-offset2)*(freq2*20 + 20)) # + sine curve 2
    series += 0.1*(np.random.rand(batch_size, n_steps)-0.5) # + bias
    return series[..., np.newaxis].astype(np.float32)

## generate datasets of train, validation, test
n_steps= 50
series= generate_time_series(10000, n_steps+1)
X_train, y_train= series[:7000, :n_steps], series[:7000, -1]
X_valid, y_valid= series[7000:9000, :n_steps], series[7000:9000, -1]
X_test, y_test= series[9000:, :n_steps], series[9000:, -1]

print('shape of datasets : batch_size {} / n_steps {} / dimensionality {}'.\
    format(series.shape[0], series.shape[1], series.shape[2]))
print('shape of X_train : \
    batch_size {} / n_steps {} / dimensionality {}'.\
    format(X_train.shape[0], X_train.shape[1], X_train.shape[2]))
print('shape of y_train : batch_size {} / dimensionality {}'.\
    format(y_train.shape[0], y_train.shape[1]))


################################################################################
################################################################################
## simple Dense
model= tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=list(X_train.shape[-2:])),
    tf.keras.layers.Dense(1)
    ])

model.compile(loss=tf.losses.MeanSquaredError(),
            optimizer=tf.optimizers.Adam(),
            metrics=[tf.metrics.MeanAbsoluteError()])

EPOCHS=20
history = model.fit(X_train, y_train, validation_split=0.25, epochs=EPOCHS, verbose=1)

## visualize results of training 
def visualize_result(result, val_result):
    plt.plot(history.history[result])
    plt.plot(history.history[val_result])
    plt.title(str(result))
    plt.xlabel('Epoch')
    plt.ylabel(str(result))
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.ylim(0,1)
    plt.show()

visualize_result('mean_absolute_error', 'val_mean_absolute_error')
visualize_result('loss', 'val_loss')


## Build Simple RNN
model= tf.keras.Sequential([
    ## Basically, SimpleRNN layer uses 'Hyperbolic tangent function' as activation.
    tf.keras.layers.SimpleRNN(1, input_shape=[None, 1]) 
    ])
model.compile(loss=tf.losses.MeanSquaredError(),
            optimizer=tf.optimizers.Adam(),
            metrics=[tf.metrics.MeanAbsoluteError()])

EPOCHS=20
history = model.fit(X_train, y_train, validation_split=0.25, epochs=EPOCHS, verbose=1)
visualize_result('mean_absolute_error', 'val_mean_absolute_error')
visualize_result('loss', 'val_loss')


## Build Deep RNN
model= tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]), 
    tf.keras.layers.SimpleRNN(20, return_sequences=True), 
    tf.keras.layers.SimpleRNN(1) 
    ])
model.compile(loss=tf.losses.MeanSquaredError(),
            optimizer=tf.optimizers.Adam(),
            metrics=[tf.metrics.MeanAbsoluteError()])

EPOCHS=20
history = model.fit(X_train, y_train, validation_split=0.25, epochs=EPOCHS, verbose=1)
visualize_result('mean_absolute_error', 'val_mean_absolute_error')
visualize_result('loss', 'val_loss')


## Deep RNN model 2
model= tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]), 
    tf.keras.layers.SimpleRNN(20), 
    tf.keras.layers.Dense(1) 
    ])
model.compile(loss=tf.losses.MeanSquaredError(),
            optimizer=tf.optimizers.Adam(),
            metrics=[tf.metrics.MeanAbsoluteError()])

EPOCHS=20
history = model.fit(X_train, y_train, validation_split=0.25, epochs=EPOCHS, verbose=1)
visualize_result('mean_absolute_error', 'val_mean_absolute_error')
visualize_result('loss', 'val_loss')




################################################################################
################################################################################
## multi-step prediction model
## using for loop and single-step prediction model
series= generate_time_series(1, n_steps + 10)
X_new, Y_new= series[:, :n_steps], series[:, n_steps:]
X= X_new
for step_ahead in range(10):
    y_pred_one= model.predict(X[:, step_ahead:])[:, np.newaxis, :]
    X= np.concatenate([X, y_pred_one], axis=1)

y_pred= X[:, n_steps:]

history= np.squeeze(series)[:n_steps]
y_new= np.squeeze(series)[n_steps:]
y_graph_pred= np.squeeze(y_pred)

def visaulize_pred(n_steps, history, y_new, y_graph_pred):
    plt.plot(np.arange(n_steps), history)
    plt.plot(np.arange(n_steps, n_steps+10), y_new, color='blue')
    plt.plot(np.arange(n_steps, n_steps+10), y_graph_pred, color='red')
    plt.plot(np.arange(n_steps, n_steps+10), y_new, 'bx', label='real')
    plt.plot(np.arange(n_steps, n_steps+10), y_graph_pred, 'ro', label='pred')
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.grid()
    plt.legend()
    plt.show()
    



################################################################################
################################################################################
## multi-step RNN model
n_steps= 50
series= generate_time_series(10000, n_steps+10)
X_train, y_train= series[:7000, :n_steps], series[:7000, -10:, 0]
X_valid, y_valid= series[7000:9000, :n_steps], series[7000:9000, -10, 0]
X_test, y_test= series[9000:, :n_steps], series[9000:, -10, 0] 

model= tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]), 
    tf.keras.layers.SimpleRNN(20), 
    tf.keras.layers.Dense(10) 
    ])
model.compile(loss=tf.losses.MeanSquaredError(),
            optimizer=tf.optimizers.Adam(),
            metrics=[tf.metrics.MeanAbsoluteError()])
y_pred= model.predict(X_new)

EPOCHS=20
history = model.fit(X_train, y_train, validation_split=0.25, epochs=EPOCHS, verbose=1)

visualize_result('mean_absolute_error', 'val_mean_absolute_error')
visualize_result('loss', 'val_loss')



series= generate_time_series(1, n_steps + 10)
X_new, Y_new= series[:, :n_steps], series[:, n_steps:]
y_pred= model.predict(X_new)

history= np.squeeze(X_new)
y_new= np.squeeze(Y_new)
y_graph_pred= np.squeeze(y_pred)
visaulize_pred(n_steps, history, y_new, y_graph_pred)

m = tf.keras.metrics.RootMeanSquaredError()
m.update_state(list(y_new), list(y_graph_pred))
m.result().numpy()

################################################################################
################################################################################
## Timedistributed layer
Y= np.empty((10000, n_steps, 10))
for step_ahead in range(1, 10+1):
    Y[:, :, step_ahead-1]= series[:, step_ahead:step_ahead+n_steps, 0]

Y_train= Y[:7000]
Y_valid= Y[7000:9000]
Y_test= Y[9000:]

model= tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]), 
    tf.keras.layers.SimpleRNN(20, return_sequences=True), 
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(10))
    ])

def last_time_step_mse(Y_true, Y_pred):
    return tf.keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])

model.compile(loss=tf.losses.MeanSquaredError(),
            optimizer=tf.optimizers.Adam(learning_rate=0.01),
            metrics=[last_time_step_mse])

EPOCHS=20
history = model.fit(X_train, Y_train, validation_split=0.25, epochs=EPOCHS, verbose=1)
history.history
visualize_result('last_time_step_mse', 'val_last_time_step_mse')
visualize_result('loss', 'val_loss')
