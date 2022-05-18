from math import sqrt
from numpy import concatenate
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model

from load_data import load_data_from_csv


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


real_y = []
predicted_y = []


for i in range(40,80):
    train_X, train_y, vali_X, vali_y, test_X, test_y, scaler = load_data_from_csv(csv_file_name='train_data.csv',
                                                                                  train_vali_split=688,
                                                                                  vali_test_split=835 + i,
                                                                                  test_set_size=1)
    i_th_model = load_model('models/best_test_model_for_idx_'+str(i)+'.h5')
    _, i_th_real_y, i_th_predicted_y = evaluate_model_rmse(model=i_th_model, input_X=test_X, label_y=test_y, scaler=scaler)
    i_th_real_y = i_th_real_y[0]
    i_th_predicted_y = i_th_predicted_y[0]
    if i_th_predicted_y < 0.0:
        i_th_predicted_y = 0.0
    real_y.append(i_th_real_y)
    predicted_y.append(i_th_predicted_y)

rmse = sqrt(mean_squared_error(real_y, predicted_y))
print('Test RMSE: %.3f' % rmse)
