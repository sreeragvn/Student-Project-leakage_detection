# %%
from utils.data_preprocess import load_single_leakage_model_data
import tensorflow as tf
from tensorflow import keras
import numpy as np

# %%
residual_subtract = False
augmentation = False
blind_flip = False
tot_flow = False
tot_resflow = False
res_flow = False

# %%
# model = tf.keras.models.load_model('saved_model/single_leakage_model_new')
# model.summary()

# %%
X_train, X_test, X_val, y_train, y_test, y_val, scaler_coords, scaler_flows  = load_single_leakage_model_data(residual_subtract, augmentation, 
                                                                                                    blind_flip, tot_flow, 
                                                                                                    res_flow, tot_resflow)

# %%
import pandas as pd
results_org = pd.DataFrame(columns=['start_learning_rate', 'width', 'depth', 'l2_weight','l1_weight', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])

epoch = 10
start_learning_rates = [1e-3, 1e-2, 1e-4]
widths_low = [2, 4, 8, 16]
widths_high = list(range(32, 512, 32))
widths = widths_high + widths_low
depths = list(range(1, 16,1))
l2_weights = [0, 1e-1, 1e-2, 1e-3]
l1_weights = [0, 1e-1, 1e-2, 1e-3]

end_learning_rate = 1e-8
decay_steps = 10000
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

# results = pd.DataFrame()
results_path = "results_2.csv"
results_org.to_csv(results_path)
for start_learning_rate in start_learning_rates:
    for width in widths:
        for depth in depths:
            for l2_weight in l2_weights:
                for l1_weight in l1_weights:
                    results = pd.read_csv(results_path, index_col=0)
                    model = keras.models.Sequential()
                    for _ in range(depth):
                        model.add(keras.layers.Dense(units=width, activation="relu", 
                                                     kernel_regularizer=keras.regularizers.L1L2(l1 = l1_weight, l2 = l2_weight),
                                                     kernel_initializer='he_uniform'))
                    model.add(tf.keras.layers.Dense(2,activation= "linear", kernel_initializer='he_uniform'))
                    scheduler = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate= start_learning_rate,
                                                                              decay_steps= decay_steps, 
                                                                              end_learning_rate= end_learning_rate,
                                                                              power=1)
                    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=scheduler),
                                  loss="mse",
                                  metrics='mae')
                    history = model.fit(X_train, y_train, epochs=epoch, 
                                        validation_data = (X_val, y_val), 
                                        callbacks= stop_early,
                                        shuffle= True)
                    train_loss, train_acc = model.evaluate(X_train, y_train)
                    val_loss, val_acc = model.evaluate(X_val, y_val)
                    results_tmp = np.array([start_learning_rate, width, depth, l2_weight, l1_weight, train_loss, val_loss, train_acc, val_acc], dtype=object).reshape(-1,1)
                    results = np.hstack((results_tmp, np.array(results, dtype=object).transpose())).transpose()
                    results = pd.DataFrame(data=results, columns=results_org.columns)
                    results.to_csv(results_path)
