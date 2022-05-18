from math import sqrt
from numpy import concatenate
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam


def creare_model(number_lstm_neurons=100,dense_layer=False,decay_rate=0.00014):
    # design network
    model = Sequential()
    model.add(LSTM(number_lstm_neurons, input_shape=(1, 28)))
    if dense_layer:
        model.add(Dense(30, activation='relu'))
    model.add(Dense(1))
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False, decay=decay_rate)
    model.compile(loss='mae', optimizer=adam, metrics=['accuracy'])
    return model

