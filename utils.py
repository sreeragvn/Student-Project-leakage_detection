import os
import itertools
# import sys 
# sys.path.append('/media/vn/Data/Workspace/leakage_detection_cfrp')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data_acquisition.wing_leakage_data_acquisition_aux import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# Libraries
import numpy as np
# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__)
from tensorflow import keras
from keras import layers
import keras_tuner as kt

np.random.seed(0) 
tf.random.set_seed(0)

print("# GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
tf.keras.backend.set_floatx('float64')
tf.compat.v1.enable_eager_execution()

def load_data(total_samples):
    data = pd.read_csv('./data_acquisition/wing_leakage_data_samples_filt_bad_out.csv')
    data = data.drop(data[data.quality == 'bad'].index)
    # data = data[data['MFC9'].notna()]
    data = data.drop(data[data.sample_number > total_samples].index)
    data = data.drop(data[data.Comments == 'outside'].index)
    data = data.drop(data[data.Comments == 'Repeat'].index)
    data = data.drop(data[data.Comments == 'repeat'].index)
    data = data.drop(data[data.Comments == 'on port'].index)
    data = data.drop(data[data.Comments == 'missed'].index)
    data = data.drop(data[data.Comments == 'Not sure if I will be able to seal the leakage on y axis'].index)
    data = data.drop(columns=['sample_number', 'total flow rate','Comments','Day'])
    data = data.drop(columns=['quality'])
    data = data.rename(columns={'number of leakage':'number_of_leakage'})
    # data.columns
    single_leakage = data[data['number_of_leakage'] == 1]
    single_leakage = single_leakage.drop(columns=['x2', 'y2'])
    two_leakage = data[data['number_of_leakage'] == 2]

    single_leakage = single_leakage.drop(columns=['number_of_leakage'])
    two_leakage = two_leakage.drop(columns=['number_of_leakage'])
    # single_leakage.to_csv("single_leakage.csv")   
    # print(single_leakage.shape, two_leakage.shape)
    return single_leakage, two_leakage

def flipped_data(data_x, data_y, blind):
    #code applicable for single leakage case 
    data = data_x.copy()

    data_y = data_y.copy()
    y_flip = np.subtract(5200.0, data_y.transpose()[1])
    x = data_y.transpose()[0]
    data_y = np.vstack((x, y_flip)).transpose()

    temp1 = data.transpose()[0]
    temp2 = data.transpose()[1]
    temp3 = data.transpose()[2]
    temp4 = data.transpose()[3]
    temp5 = data.transpose()[4]
    temp6 = data.transpose()[5]
    temp7 = data.transpose()[6]
    temp8 = data.transpose()[7]
    temp9 = data.transpose()[8]
    temp10 = data.transpose()[9]

    data = np.vstack((temp6, temp7, temp8,
                      temp9, temp10, temp1,
                      temp2, temp3, temp4,
                      temp5)).transpose()
    
    if blind == False:
        data_y = pd.DataFrame(data_y, columns=['x1', 'y1'])
        data = pd.DataFrame(data)
        index = data_y.index[data_y['x1'] <= 2650].tolist()
        # print(index)
        data_y = data_y.drop(index=index)
        data = data.drop(index=index)

    return data, data_y

def data_check(data):
    data_plot(data)
    print(data.isna().any())
    data.describe().transpose()
    sns.pairplot(data[data.columns.values], diag_kind='kde')

def model_eval(model, X_test, y_test, scaler_coords, X_train, y_train, X_val, y_val, history):
    y_predictions = model.predict(X_test)
    # print("test", "{:10.4f}".format(mean_squared_error(y_test, y_predictions, squared=True)))
    # plot_test_pred(y_test, y_predictions, scaler_coords)
    results_train = model.evaluate(X_train, y_train)
    results_val = model.evaluate(X_val, y_val)
    results_test = model.evaluate(X_test, y_test)
    results = list(itertools.chain(results_train, results_val, results_test))
    results = ["{:10.4f}".format(x) for x in results]
    # plot_loss(history)
    # model.summary()
    # # plot_predictions(y_test, y_predictions)
    # loss_val = "{:10.4f}".format(results_val[0])
    # metric_val = "{:10.4f}".format(results_val[1])
    # loss_test = "{:10.4f}".format(results_test[0])
    # metric_test = "{:10.4f}".format(results_test[1])
    # return loss_val, metric_val, loss_test, metric_test
    return results
    # error_x_histogram(y_test, y_predictions)
    # error_y_histogram(y_test, y_predictions)

def split_xy_save(single_leakage):
    x,y = training_data_xy(single_leakage)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1) # 0.25 x 0.8 = 0.2

    model_data = [X_train, X_test, y_train, y_test, X_val, y_val]
    file_name = ['X_train', 'X_test', 'y_train', 'y_test', 'X_val', 'y_val']

    for data, file_name in zip(model_data, file_name):
        data.to_csv("./model_data/" + file_name + ".csv")

def mfc_sum_scaler(flow_data):
    # data['mfcsum'] = data['MFC1'] + data['MFC2'] + data['MFC3'] + data['MFC4'] + data['MFC5'] + data['MFC6'] + data['MFC7'] + data['MFC8'] + data['MFC9'] + data['MFC10']
    # data[['MFC1', 'MFC2', 'MFC3', 'MFC4', 'MFC5', 'MFC6', 'MFC7', 'MFC8', 'MFC9', 'MFC10']] = data[['MFC1', 'MFC2', 'MFC3', 'MFC4', 'MFC5', 'MFC6', 'MFC7', 'MFC8', 'MFC9', 'MFC10']].div(data['mfcsum'], axis=0)
    # mfc_sum = data['mfcsum']
    # data = data.drop(columns=['mfcsum'])

    mfc_sum = np.sum(flow_data, axis=1)
    flow_data = flow_data / mfc_sum[:,None]
    return flow_data, mfc_sum

def normalize_data(X_train, X_test, X_val, y_train, y_test, y_val):

    if 'x2' in y_train.columns:
        scaler_coords = StandardScaler()
        y_train = scaler_coords.fit_transform(y_train)
        y_test = scaler_coords.transform(y_test)
        y_val = scaler_coords.transform(y_val)
    else:
        scaler_coords = StandardScaler()
        y_train = scaler_coords.fit_transform(y_train)
        y_test = scaler_coords.transform(y_test)
        y_val = scaler_coords.transform(y_val)
    
    scaler_flows = StandardScaler()
    X_train = scaler_flows.fit_transform(X_train)
    X_test = scaler_flows.transform(X_test)
    X_val = scaler_flows.transform(X_val)

    return X_train, X_test, X_val, y_train, y_test, y_val, scaler_coords, scaler_flows

def scale_transform_coords(data, span, width):
    # data[:, 0:1] = data[:, 0:1]/(span)
    # data[:, 1:2] = data[:, 1:2]/(width)

    data[:, 0:1] = data[:, 0:1] * 2 - 1
    data[:, 1:2] = data[:, 1:2] * 2 - 1

    return data

def model_comparison(model_metric, model_evaluate, label, augmentation, residual_subtract, mfc_sum_scaler, blind_flip):
    results = np.array((str(label), augmentation, residual_subtract, mfc_sum_scaler, blind_flip))
    results = np.hstack((results, model_evaluate))
    model_performance = pd.DataFrame(results.reshape(-1,1).transpose(),
                                     columns=['model_name','Augmentation', 
                                              'Residual_subtract', 'mfc_sum_scaler', 'blind_flip',
                                              'loss_train', 'metric_train',
                                              'loss_val', 'metric_val', 
                                              'loss_test', 'metric_test'])
    model_metric = pd.concat((model_metric, model_performance), axis=0)
    model_metric.reset_index(drop=True, inplace=True)
    return model_metric

def load_single_leakage_model_data(aug, residual_subtract, mfc_sum_scaler, blind):
    X_train = pd.read_csv('./model_data/X_train.csv', index_col=0)
    X_test = pd.read_csv('./model_data/X_test.csv', index_col=0)
    X_val = pd.read_csv('./model_data/X_val.csv', index_col=0)

    y_train = pd.read_csv('./model_data/y_train.csv', index_col=0).to_numpy()
    y_test = pd.read_csv('./model_data/y_test.csv', index_col=0)
    y_val = pd.read_csv('./model_data/y_val.csv', index_col=0)
    
    x = [X_train, X_val, X_test]
    if residual_subtract is True:
        for data in x:
            data['MFC10'] = data['MFC10'] - data['mfc10_residual']

    X_train = X_train.drop(columns=['mfc10_residual'])
    X_test = X_test.drop(columns=['mfc10_residual'])
    X_val = X_val.drop(columns=['mfc10_residual'])

    X_train = X_train.to_numpy()

    if aug == True:
        X_train_flip, y_train_flip = flipped_data(X_train, y_train, blind)
        # X_train_flip = X_train_flip.drop(columns=['mfc10_residual'])
        # print(X_train_flip.shape, y_train_flip.shape)
        # print(X_train.shape, y_train.shape)
        X_train = np.vstack((X_train, X_train_flip))
        y_train = np.vstack((y_train, y_train_flip))
        print("augmented data is loaded")

    X_train = pd.DataFrame(X_train, columns=X_test.columns)
    y_train = pd.DataFrame(y_train, columns=y_test.columns)

    # y_train = dim_scaler(y_train, span, width,)

    if mfc_sum_scaler == True:
        X_train, train_flow_sum = mfc_sum_scaler(X_train)
        X_test, test_flow_sum = mfc_sum_scaler(X_test)
        X_val, val_flow_sum = mfc_sum_scaler(X_val)

    X_train, X_test, X_val, y_train, y_test, y_val, scaler_coords, scaler_flows = normalize_data(X_train, X_test, X_val, y_train, y_test, y_val)

    # print(X_train.shape, X_test.shape, X_val.shape, y_train.shape, y_test.shape, y_val.shape)
    # y_train = scale_transform_coords(y_train, span, width)
    # y_test = scale_transform_coords(y_test, span, width)
    # y_val = scale_transform_coords(y_val, span, width)

    return X_train, X_test, X_val, y_train, y_test, y_val, scaler_coords, scaler_flows

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    # plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()

def plot_predictions(y_test, y_predictions):
    a = plt.axes(aspect='equal')
    plt.scatter(y_test[:,0:1], y_predictions[:,0:1], label='x')
    plt.scatter(y_test[:,1:2], y_predictions[:,1:2], label='y')
    plt.xlabel('True Values')
    plt.ylabel('Predictiond')
    plt.legend()
    lims = [-1, 1]
    plt.tight_layout()
    plt.xlim(lims)
    plt.ylim(lims)
    plt.tight_layout()
    plt.plot(lims, lims)
    plt.show()

def error_x_histogram(y_test, y_predictions):
    error_x = y_test['x1'].values.reshape(-1,1) - y_predictions[:,0:1]
    plt.hist(error_x, bins=25)
    plt.xlabel('Prediction Error X')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

def error_y_histogram(y_test, y_predictions):
    erroy_y = y_test['y1'].values.reshape(-1,1) - y_predictions[:,1:2]
    plt.hist(erroy_y, bins=25)
    plt.xlabel('Prediction Error y')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()


#Define the model using model builder from keras tuner
def model_builder(hp):
    model = keras.Sequential()

    # Choose an optimal value between 32-512
    for i in range(hp.Int("num_layers", 1, 15)):
        model.add(
            keras.layers.Dense(
                units=hp.Int("units_" + str(i), min_value=32, max_value=512, step=32),
                activation=hp.Choice("activation", ["relu", "tanh", "elu"]),
        )
    )
    if hp.Boolean("dropout"):
        model.add(layers.Dropout(rate=0.1))
    model.add(keras.layers.Dense(units=2, activation= "linear"))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    # hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-1, sampling="log")

    # model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    #             loss="mse",  metrics='mae')
    
    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=learning_rate),
                loss="mse",
                # loss = 'mean_squared_logarithmic_error',
                metrics='mae')

    return model

#search for the best hyperparameters and train the standard model with original training data
def hyper_model(X_train,Y_train, X_val, y_val, EPOCH, factor, folder_name):
    tuner = kt.Hyperband(model_builder,
                         objective='val_loss',
                        # objective='val_mean_absolute_error',
                         max_epochs=EPOCH,
                         factor=factor,
                         hyperband_iterations = 1,
                        # Integer, at least 1, the number of times to iterate over the full Hyperband algorithm. One iteration will 
                        # run approximately max_epochs * (math.log(max_epochs, factor) ** 2) cumulative epochs across all trials. It is 
                        # recommended to set this to as high a value as is within your resource budget. Defaults to 1.
                         directory="../tensorflow_log_files/studienarbeit/",
                         seed=0,    
                         project_name=str(folder_name))

    tuner.search_space_summary()
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    tuner.search(X_train, Y_train, epochs=EPOCH, validation_data = (X_val, y_val), callbacks=[stop_early, 
                                                                                              keras.callbacks.TensorBoard("../tensorflow_log_files/studienarbeit/tb_logs"+str(folder_name))
                                                                                              ])
    #tuner.search(X_train, Y_train, epochs=50, validation_data=(X_test,Y_test), callbacks=[stop_early])
    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # print(f"""
    # The hyperparameter search is complete. The optimal learning rate for the optimizer
    # is {best_hps.get('learning_rate')}.
    # """)

    model = tuner.hypermodel.build(best_hps)
    history = model.fit(X_train, Y_train, epochs=EPOCH, validation_data = (X_val, y_val), shuffle= False)

    return best_hps, model, tuner, history


def data_plot(data):
    # plt.figure(figsize=(8, 6), dpi=80)
    fig, axs = plt.subplots(len(data.columns),1)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    # fig.tight_layout()
    axs = axs.flatten()
    # axs[1].plot(single_leakage['MFC1'])
    colname = data.columns.values
    for i,col in enumerate(colname):
            ax = axs[i]
            ax.plot(data[col])
        
    # Set common labels
    # fig.title("1000 row data value")
    fig.text(0.5, 0.04, 'data', ha='center', va='center')
    fig.text(0.06, 0.5, '    value', ha='center', va='center', rotation='vertical')

    # fig.savefig("data plot.jpg")
    # ax = axs[-1]
    # ax.plot(tf.reduce_sum(data, axis=1))

def variance_plot(data):
    mean =tf.reduce_mean(data, axis=0)
    std =tf.math.reduce_std(data, axis=0)
    # print("Mean of inputs: {}".format(mean[2:12]))
    
    x = np.array([6, 7, 8, 9, 10, 1, 2, 3, 4, 5])
    y = mean[2:12]
    e = std[2:12]
    plt.boxplot(data.drop(columns=['x1', 'y1']))
    # plt.errorbar(x, y, e, linestyle='None', marker='^')
    plt.show()

def training_data_xy(data):
    y = data[['x1', 'y1']] 
    x = data.drop(['x1', 'y1'], axis=1)
    return x,y

def single_leakage_data_spread(single_leakage):
    
    fig = plt.figure(figsize=(12, 4), dpi=80)
    # Create data: 200 points
    data =  single_leakage[['x1','y1']]
    x, y = single_leakage['x1'].T,single_leakage['y1'].T

    # span = 16048
    # width = 5233

    # plt.plot([0, 0.494142572], [0, 0], 'k')
    # plt.plot([7930, 16048], [0, 1149], 'k')
    # plt.plot([16048, 16048], [1149, 4386], 'k')
    # plt.plot([16048, 7843], [4386, 5233], 'k')
    # plt.plot([7843, 2493], [5233, 5233], 'k')
    # plt.plot([2493, 0], [5233, 0], 'k')

    nbins = 20
    k = gaussian_kde(data.T)
    xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    
    # contour
    plt.title('Spread of leakage made across the wing')
    # plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGns_r)
    plt.contour(xi, yi, zi.reshape(xi.shape) )
    cp = plt.contourf(xi, yi, zi.reshape(xi.shape))
    plt.colorbar(cp)
    fig1 = plt.gcf()
    plt.tight_layout()
    plt.show()
    # fig1.savefig('plot.png', dpi=100)

def plot_leakages(mfc):
    plt.figure(figsize=(40, 20))
    
    # plt.title(f'Sample Number {sample_number}', fontsize=20)
    
    # plot sensor positions
    sensors = np.array([[2426, 70], [5480, 70], [8661, 191], [11676, 584], [13976, 917], [2603, 5163], [5723, 5163], [8417, 5103], [11646, 4740], [14641, 4391]])
    for i in range(len(sensors)):
        plt.scatter(sensors[i, 0], sensors[i, 1], color='tab:blue', s=100)
        if i < 5:
            plt.text(sensors[i, 0], sensors[i, 1] - 200, str(i+1), fontsize='xx-large')
        else:
            plt.text(sensors[i, 0], sensors[i, 1] + 350, str(i+1), fontsize='xx-large')

    # plot leakage positions
    plt.scatter(X, Y, color='black', s=10)
    
    plt.scatter(mfc['x1'], mfc['y1'], color='tab:red', s=200)
    
    # print(X.shape)
    

    # plot wing contour
    plot_wing_contour()

    # include grid coordinate system
    plt.hlines(0, -1000, 17000, linestyle='dashed')
    plt.hlines(2600, -1000, 17000, linestyle='dashed')
    plt.hlines(5233, -1000, 17000, linestyle='dashed')
    plt.vlines(0, -1000, 6000, linestyle='dashed')
    plt.vlines(7930, -1000, 6000, linestyle='dashed')
    plt.vlines(16048, -1000, 6000, linestyle='dashed')
    plt.text(-850, -75, '$y = 0$', fontsize=20)
    plt.text(-850, 2600-75, '$y = 2600$', fontsize=20)
    plt.text(-850, 5233-75, '$y = 5233$', fontsize=20)
    plt.text(75, -700, '$x=0$', fontsize=20)
    plt.text(7930+75, -700, '$x=7930$', fontsize=20)
    plt.text(16048+75, -700, '$x=16048$', fontsize=20)
    # plt.text(180, 5800, f'(x1, y1) = ({x1}, {y1}) = ({j1-31}, {-i1+10})', fontsize=20)

    # invert y axis
    plt.gca().invert_yaxis()

    plt.show()
    
# meshgrid to coords conversion
def meshgrid_to_coords(i,j):
    x = X[-j - 10, i+31]
    y = Y[-j - 10, i+31]
    print(x,y)

def get_leakage_close_to_sensor():
    # data = pd.read_excel('./data_acquisition/wing_leakage_data_samples_filt.xlsx')
    data = pd.read_csv('./data_acquisition/wing_leakage_data_samples_filt_bad_out.csv')
    data = data.drop(data[data.quality == 'bad'].index)
    data = data.drop(columns=['quality'])
    single_leakage = data.rename(columns={'number of leakage':'number_of_leakage'})
    single_leakage = single_leakage[single_leakage['number_of_leakage'] == 1]
    flows = single_leakage.drop(columns=['sample_number', 'total flow rate', 'x1', 'y1', 'x2', 'y2',
                                                'Comments', 'Day', 'number_of_leakage'])
    single_leakage = single_leakage.drop(columns=['sample_number', 'total flow rate','x2', 'y2',
                                                'Comments', 'Day', 'number_of_leakage'])
    mfc1 = single_leakage.loc[(single_leakage['MFC1'] == flows.max(axis=1))]
    mfc2 = single_leakage.loc[(single_leakage['MFC2'] == flows.max(axis=1))]
    mfc3 = single_leakage.loc[(single_leakage['MFC3'] == flows.max(axis=1))]
    mfc4 = single_leakage.loc[(single_leakage['MFC4'] == flows.max(axis=1))]
    mfc5 = single_leakage.loc[(single_leakage['MFC5'] == flows.max(axis=1))]
    mfc6 = single_leakage.loc[(single_leakage['MFC6'] == flows.max(axis=1))]
    mfc7 = single_leakage.loc[(single_leakage['MFC7'] == flows.max(axis=1))]
    mfc8 = single_leakage.loc[(single_leakage['MFC8'] == flows.max(axis=1))]
    mfc9 = single_leakage.loc[(single_leakage['MFC9'] == flows.max(axis=1))]
    mfc10 = single_leakage.loc[(single_leakage['MFC10'] == flows.max(axis=1))]

    return data, mfc1, mfc2, mfc3, mfc4, mfc5, mfc6, mfc7, mfc8, mfc9, mfc10

def get_index_anomaly(mfc, coord, level, less):
    if not less:
        return mfc.loc[(mfc[coord] >= level)]
    else:
        return mfc.loc[(mfc[coord] <= level)]
    
def plot_test_pred(test, pred, scaler_coords):
    # pred = pred * test_flow_sum[:,None]
    # test = test * test_flow_sum[:,None]

    # pred[:, 0:1] = (pred[:, 0:1] + 1) / 2
    # pred[:, 1:2] = (pred[:, 1:2] + 1) / 2
    # test[:, 0:1] = (test[:, 0:1] + 1) / 2
    # test[:, 1:2] = (test[:, 1:2] + 1) / 2

    pred = scaler_coords.inverse_transform(pred)
    test = scaler_coords.inverse_transform(test)
    plt.figure(figsize=(40, 20))
    
    # plt.title(f'Sample Number {sample_number}', fontsize=20)
    
    # plot sensor positions
    sensors = np.array([[2426, 70], [5480, 70], [8661, 191], [11676, 584], [13976, 917], [2603, 5163], [5723, 5163], [8417, 5103], [11646, 4740], [14641, 4391]])
    for i in range(len(sensors)):
        plt.scatter(sensors[i, 0], sensors[i, 1], color='tab:blue', s=100)
        if i < 5:
            plt.text(sensors[i, 0], sensors[i, 1] - 200, str(i+1), fontsize='xx-large')
        else:
            plt.text(sensors[i, 0], sensors[i, 1] + 350, str(i+1), fontsize='xx-large')

    # plot leakage positions
    plt.scatter(X, Y, color='black', s=10)
    
    for i in range(len(test)):
        if i != len(test)-1:
            plt.scatter(test[i][0], test[i][1], color='tab:red', s=200)
            plt.scatter(pred[i][0], pred[i][1], color='tab:blue', s=200)
            line = np.vstack((test[i], pred[i])).transpose()
            plt.plot(line[0], line[1], color = 'black')
        else:
            plt.scatter(test[i][0], test[i][1], color='tab:red', s=200, label="true")
            plt.scatter(pred[i][0], pred[i][1], color='tab:blue', s=200, label="pred")
            line = np.vstack((test[i], pred[i])).transpose()
            plt.plot(line[0], line[1], color = 'black')

    # print(X.shape)

    # plot wing contour
    plot_wing_contour()

    # include grid coordinate system
    plt.hlines(0, -1000, 17000, linestyle='dashed')
    plt.hlines(2600, -1000, 17000, linestyle='dashed')
    plt.hlines(5233, -1000, 17000, linestyle='dashed')
    plt.vlines(0, -1000, 6000, linestyle='dashed')
    plt.vlines(7930, -1000, 6000, linestyle='dashed')
    plt.vlines(16048, -1000, 6000, linestyle='dashed')
    plt.text(-850, -75, '$y = 0$', fontsize=20)
    plt.text(-850, 2600-75, '$y = 2600$', fontsize=20)
    plt.text(-850, 5233-75, '$y = 5233$', fontsize=20)
    plt.text(75, -700, '$x=0$', fontsize=20)
    plt.text(7930+75, -700, '$x=7930$', fontsize=20)
    plt.text(16048+75, -700, '$x=16048$', fontsize=20)
    # plt.text(180, 5800, f'(x1, y1) = ({x1}, {y1}) = ({j1-31}, {-i1+10})', fontsize=20)
    plt.legend(loc="upper left")
    # invert y axis
    plt.gca().invert_yaxis()
    plt.savefig('./results/hypermodel_results.png')

    plt.show()

def benchmark_linear_model(X_train, y_train, X_val, y_val):
    linear_model = keras.Sequential([layers.Dense(2)])
    # linear_model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.01), metrics='mae')
    linear_model.compile(
                        loss='mean_squared_error',
                        # loss= 'mean_squared_logarithmic_error',
                        optimizer=tf.keras.optimizers.Nadam(0.01), metrics='mae')
    
    # linear_model.summary()

    # %%time
    history = linear_model.fit(
        X_train,
        y_train,
        epochs=100,
        # Suppress logging.
        verbose=0,
        validation_data = (X_val, y_val))
    
    return linear_model, history

def scikit_linear_regression(X_train, y_train, X_test, y_test, scaler_coords, X_val, y_val):
    reg = LinearRegression().fit(X_train, y_train)
    y_predictions_train = reg.predict(X_train)
    # print("train", "{:10.4f}".format(mean_squared_error(y_train, y_predictions, squared=True)))
    y_predictions_val = reg.predict(X_val)
    # print("val", "{:10.4f}".format(mean_squared_error(y_val, y_predictions, squared=True)))
    y_predictions = reg.predict(X_test)
    loss_test = "{:10.4f}".format(mean_squared_error(y_test, y_predictions, squared=True))
    metric_test = "{:10.4f}".format(mean_absolute_error(y_test, y_predictions))

    loss_val = "{:10.4f}".format(mean_squared_error(y_val, y_predictions_val, squared=True))
    metric_val = "{:10.4f}".format(mean_absolute_error(y_val, y_predictions_val))

    loss_train = "{:10.4f}".format(mean_squared_error(y_train, y_predictions_train, squared=True))
    metric_train = "{:10.4f}".format(mean_absolute_error(y_train, y_predictions_train))

    results = [loss_train, metric_train, loss_val, metric_val, loss_test, metric_test]
    # print(results)
    results = [float(x) for x in results]
    return results