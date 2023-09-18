import os
import itertools
# import sys 
# sys.path.append('/media/vn/Data/Workspace/leakage_detection_cfrp')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")
import yaml

# Libraries
import numpy as np
# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)
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

with open("config.yml", "r") as ymlfile:
    cfg = yaml.full_load(ymlfile)

def model_eval(model, X_test, y_test, X_train, y_train, X_val, y_val):
    y_predictions = model.predict(X_test)
    results_train = model.evaluate(X_train, y_train, verbose=1)
    results_val = model.evaluate(X_val, y_val, verbose=1)
    results_test = model.evaluate(X_test, y_test, verbose=1)
    results = list(itertools.chain(results_train, results_val, results_test))
    results = ["{:10.4f}".format(x) for x in results]
    return results, y_predictions

def numpy_to_tensor(X_train, X_test, X_val, y_train, y_test, y_val):
    X_train = tf.convert_to_tensor(X_train, np.float32)
    X_test = tf.convert_to_tensor(X_test, np.float32)
    X_val = tf.convert_to_tensor(X_val, np.float32)
    y_train = tf.convert_to_tensor(y_train, np.float32)
    y_test = tf.convert_to_tensor(y_test, np.float32)
    y_val = tf.convert_to_tensor(y_val, np.float32)
    return X_train, X_test, X_val, y_train, y_test, y_val

def model_comparison(model_metric, model_evaluate, label, augmentation, residual_subtract, blind_flip, tot_flow, res_flow, tot_resflow):
    results = np.array((str(label), augmentation, residual_subtract, blind_flip, tot_flow, res_flow, tot_resflow), dtype=object)
    results = np.hstack((results, np.array(model_evaluate, dtype=object)))
    # print(results.shape)
    model_performance = pd.DataFrame(results.reshape(-1,1).transpose(),
                                     columns=cfg['model_performance']['column_metrics'])
    model_metric = pd.concat((model_metric, model_performance), axis=0)
    model_metric.reset_index(drop=True, inplace=True)
    return model_metric

#Define the model using model builder from keras tuner
def model_builder_single_dropout(hp):
    model = keras.Sequential()

    # Choose an optimal value between 32-512
    for i in range(hp.Int("num_layers", 1, 15)):
        model.add(
            keras.layers.Dense(
                units=hp.Int("units_" + str(i), min_value=32, max_value=512, step=32),
                activation=hp.Choice("activation", ["relu", "tanh"]),
                kernel_initializer='he_uniform'
            )
        )
        model.add(layers.Dropout(rate=0.1))
    # if hp.Boolean("dropout"):
    #   model.add(layers.Dropout(rate=0.1))
    model.add(keras.layers.Dense(units=2, activation= "linear", kernel_initializer='he_uniform'))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    # hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-1, sampling="log")

    # model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    #             loss="mse",  metrics='mae')
    
    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=learning_rate),
                loss="mse",
                metrics='mae')

    return model


#Define the model using model builder from keras tuner
def model_builder_single(hp):
    model = keras.Sequential()

    # Choose an optimal value between 32-512
    for i in range(hp.Int("num_layers", 1, 15)):
        model.add(
            keras.layers.Dense(
                units=hp.Int("units_" + str(i), min_value=32, max_value=512, step=32),
                activation=hp.Choice("activation", ["relu", "tanh", "elu"]),
                kernel_initializer='he_uniform'
            )
        )
        if hp.Boolean("dropout"):
            model.add(layers.Dropout(rate=0.1))
    model.add(keras.layers.Dense(units=2, activation= "linear", kernel_initializer='he_uniform'))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    # hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-1, sampling="log")

    # model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    #             loss="mse",  metrics='mae')
    
    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=learning_rate),
                loss="mse",
                metrics='mae')

    return model


#search for the best hyperparameters and train the standard model with original training data
def hyper_model(X_train,Y_train, X_val, y_val, epoch, factor, augmentation, residual_subtract, 
                blind_flip, tot_flow, res_flow, tot_resflow):
    folder_name = str('keras_hyperparameter_aug_'+ str(augmentation)+
                      "_res_"+ str(residual_subtract)+
                      "_blind_"+ str(blind_flip)+
                      "tot_flow"+ str(tot_flow)+
                      "res_flow"+ str(res_flow)+
                      "tot_resflow"+ str(tot_resflow)
                    )
    tuner = kt.Hyperband(model_builder_single,
                         objective='val_loss',
                         max_epochs=epoch,
                         factor=factor,
                         hyperband_iterations = 1,
                        # Integer, at least 1, the number of times to iterate over the full Hyperband algorithm. One iteration will 
                        # run approximately max_epochs * (math.log(max_epochs, factor) ** 2) cumulative epochs across all trials. It is 
                        # recommended to set this to as high a value as is within your resource budget. Defaults to 1.
                         directory="../../tensorflow_log_files/studienarbeit/",
                         seed=0,    
                         project_name=str(folder_name))

    tuner.search_space_summary()
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    tuner.search(X_train, Y_train, epochs=epoch, validation_data = (X_val, y_val), callbacks=[stop_early, 
                                                                                            #   keras.callbacks.TensorBoard("../tensorflow_log_files/studienarbeit/tb_logs"+str(folder_name))
                                                                                              ])
    #tuner.search(X_train, Y_train, epochs=50, validation_data=(X_test,Y_test), callbacks=[stop_early])
    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    model = tuner.hypermodel.build(best_hps)
    history = model.fit(X_train, Y_train, epochs=epoch, validation_data = (X_val, y_val), shuffle= False)

    print(f"""
    The hyperparameter search is complete. The optimal learning rate for the optimizer
    is {model.optimizer.lr.numpy()}.
    """)

    return best_hps, model, tuner, history

def benchmark_linear_model(X_train, y_train, X_val, y_val):
    linear_model = keras.Sequential([layers.Dense(2)])
    # linear_model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.01), metrics='mae')
    linear_model.compile(
                        loss='mean_squared_error',
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

def linear_regression(X_train, y_train, X_test, y_test, X_val, y_val):
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
    return results, y_predictions

def get_activation_functions(model):
    for i, layer in enumerate(model.layers):
        print(i, layer)
        try:
            print(" ", layer.activation)
        except AttributeError:
            print(" no activation attribute")


# Define the multi-task HyperModel
from kerastuner import HyperModel, Hyperband
class Single_leakage(HyperModel):    
    def __init__(self, input_num):
        self.input_num = input_num

    def build(self, hp):
        inputs = keras.Input(shape=(self.input_num,))
        shared_layer = inputs
        for i in range(hp.Int('num_layers', 1, 20)):
            shared_layer = layers.Dense(
                units=hp.Int("units_" + str(i), min_value=4, max_value=512, step=4),
                activation='relu',
                kernel_initializer='he_uniform'
            )(shared_layer)
        
        outputs = layers.Dense(units=2, activation= "linear", kernel_initializer='he_uniform')(shared_layer)

        model = keras.Model(inputs=inputs, outputs=outputs)
        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-1, sampling="log")

        model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=learning_rate),
                    loss="mse",
                    metrics='mae')
        return model


from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint_callback = ModelCheckpoint(
    filepath='best_single_model_tuner_weights.h5',  # Filepath to save the weights
    monitor='val_loss',               # Metric to monitor for saving
    save_best_only=True,              # Save only the best model
    save_weights_only=True,           # Save only the weights (not the full model)
    mode='min',                       # Mode to minimize the monitored metric
    verbose=1                          # Verbosity level (optional)
)

def hyper_func_model(X_train, y_train, X_val, y_val, epochs, input_num, factor):
# Create the multi-task HyperModel
    single_leakage_hypermodel = Single_leakage(input_num=input_num)

    # Define the Hyperband tuner
    tuner = Hyperband(
        single_leakage_hypermodel,
        objective='val_loss',
        max_epochs=epochs+100,
        factor=factor,
        # hyperband_iterations = 2,
        # max_retries_per_trial = 2,
        directory="../../tensorflow_log_files/studienarbeit/",
        project_name='single_leakage_final2'
    )

    tuner.search_space_summary()
    # stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    tuner.search(X_train, y_train, epochs=epochs, validation_data = (X_val, y_val), callbacks=[checkpoint_callback])
    #tuner.search(X_train, Y_train, epochs=50, validation_data=(X_test,Y_test), callbacks=[stop_early])
    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Best Hyperparameters:", best_hps)

    model = tuner.hypermodel.build(best_hps)
    history = model.fit(X_train, y_train, epochs=epochs, validation_data = (X_val, y_val), shuffle= True, callbacks=[checkpoint_callback])

    print(f"""
    The hyperparameter search is complete. The optimal learning rate for the optimizer
    is {model.optimizer.lr.numpy()}.
    """)

    return model