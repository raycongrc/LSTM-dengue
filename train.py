from math import sqrt
from numpy import concatenate
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from load_data import load_data_from_csv
from lstm_model import creare_model


import gc
gc.collect() #free memory



def train_predictor(initial_model,train_set,train_labels):
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    es = EarlyStopping(monitor='val_loss', verbose=1, patience=50)
    history = initial_model.fit(train_set, train_labels,
                        epochs=150,
                        batch_size=20,
                        validation_data=(vali_X, vali_y),
                        verbose=1,
                        shuffle=False
                        , callbacks=[es, mc]
                        )
    return initial_model, history


def evaluate_model_rmse(model,input_X,label_y,scaler):
    yhat = model.predict(input_X)
    input_X = input_X.reshape((input_X.shape[0], input_X.shape[2]))
    # invert scaling for yhat
    inv_yhat = concatenate((yhat, input_X[:, 1:7]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    # invert scaling for test_y
    label_y = label_y.reshape((len(label_y), 1))
    inv_y = concatenate((label_y, input_X[:, 1:7]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))

    return rmse , inv_y, inv_yhat

learning_rate_dacay_range = [i * 0.00001 for i in range(0,21)]
print(learning_rate_dacay_range)
best_hyperparameters_list = []
for i in range(40,80):
    train_X, train_y, vali_X, vali_y, test_X, test_y, scaler = load_data_from_csv(csv_file_name='train_data.csv',train_vali_split=755+i,vali_test_split=835+i,test_set_size=149-i)

    print(train_X.shape, train_y.shape,vali_X.shape, vali_y.shape, test_X.shape, test_y.shape)
    best_hyperparameters = []

    lowest_vali_rmse = 9999.9
    vali_iterations_count = 0
    for lrn_rate_dacay in learning_rate_dacay_range:
        for dse_lyr in [False]:
            for attempt in range(5):
                model = creare_model(number_lstm_neurons=100, decay_rate=lrn_rate_dacay,dense_layer=dse_lyr)
                model, history = train_predictor(initial_model=model,train_set=train_X,train_labels=train_y)
                vali_rmse, _ , _ = evaluate_model_rmse(model=model, input_X=vali_X, label_y=vali_y, scaler=scaler)
                if vali_rmse < lowest_vali_rmse:
                    lowest_vali_rmse = vali_rmse
                    model_file_name = "models/best_test_model_for_idx_" + str(i) + ".h5"
                    model.save(model_file_name)
                    print("Saving new best model............................................................")
                    best_hyperparameters = [i,lrn_rate_dacay,dse_lyr]
                    print("best_hyperparameters:    "+str(best_hyperparameters))
                vali_rmse = None
                model , history = None, None
                gc.collect()  # free memory
                print("Vali_iterations_count: "+str(vali_iterations_count))
                vali_iterations_count += 1

    print("Saving best hyperparameters............................................................")
    best_hyperparameters_list.append(best_hyperparameters.copy())

    print("The list of best hyperparameters is: "+str(best_hyperparameters_list))

    train_X, train_y, vali_X, vali_y, test_X, test_y, scaler = None, None, None, None, None, None, None
    best_hyperparameters, lowest_vali_rmse, vali_iterations_count = None, None, None
    gc.collect()

print("best_hyperparameters:    "+str(best_hyperparameters))