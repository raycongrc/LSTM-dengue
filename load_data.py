from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler

# convert series to supervised learning
# using this helper function, we can do the following:
# for each time step that we want to predict, put the previous n_in time steps' data
# into the same row
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# source: https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/

def load_data_from_csv(csv_file_name, train_vali_split, vali_test_split, test_set_size):
    dataset = read_csv(csv_file_name, header=0, index_col=0)
    values = dataset.values
    values = values.astype('float32')

    # read the .csv file into pandas.DataFrame object

    # normalize all the data to the range 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    # for each week, we want to predict the number of Dengue fever infections
    # according to the previous 4 weeks data
    reframed = series_to_supervised(scaled, 4, 1)

    # drop the current weeks' data, keep the total cases only
    reframed.drop(reframed.columns[[29, 30, 31, 32, 33, 34]], axis=1, inplace=True)
    print(reframed.head())

    # split into train, validation and test sets
    values = reframed.values

    train = values[:train_vali_split, :]
    vali = values[train_vali_split:vali_test_split, :]
    test = values[vali_test_split:vali_test_split + test_set_size, :]

    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    vali_X, vali_y = vali[:, :-1], vali[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    vali_X = vali_X.reshape((vali_X.shape[0], 1, vali_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    return train_X, train_y, vali_X, vali_y, test_X, test_y, scaler
