# %%
import tensorflow as tf
from utils.module import model_eval
from utils.data_preprocess import load_single_leakage_model_data
from utils.model_evaluation import plot_test_pred
import numpy as np

# Sandwiched dropout layer to the hyperparameter tuned model and did the predictions 1000 times without training the model again
# Sandwiched dropout layer to the hyperparameter tuned model and did the predictions 1000 times after training the model again
# During hyperparameter tuning, it is ensured that a dropout layer is there after a dense layer. This model is directly used for uncertainity quanitification

# %%
dropout_prob = 0.1

# %%
X_train, X_test, X_val, y_train, y_test, y_val, scaler_coords, scaler_flows = load_single_leakage_model_data(False, False,
                                                                                                              False, False)
model = tf.keras.models.load_model('saved_model/single_leakage_model')
model_evaluate, y_pred = model_eval(model, X_test, y_test, X_train, y_train, X_val, y_val)
stoch_model = tf.keras.Sequential(
    [
        model.get_layer('dense_1'),
        tf.keras.layers.Dropout(dropout_prob),
        model.get_layer('dense_2'),
        tf.keras.layers.Dropout(dropout_prob),
        model.get_layer('dense_3'),
        tf.keras.layers.Dropout(dropout_prob),
        model.get_layer('dense_4'),
    ]
    )
# stoch_model.build((1,10)) 
# stoch_model.summary()
stoch_model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=model.optimizer.lr.numpy()),
                loss="mse",
                metrics='mae')
history = stoch_model.fit(X_train, y_train, epochs=100, validation_data = (X_val, y_val), shuffle= False)
model_evaluate, y_pred = model_eval(stoch_model, X_test, y_test, X_train, y_train, X_val, y_val)

# %%
pred=np.stack([stoch_model(X_test,training=True) 
               for sample in range(1000)])

pred_mean=pred.mean(axis=0)
pred_std = pred.std(axis=0) 
print(pred_mean.shape, pred_std.shape)
# %%
plot_test_pred(y_test, pred_mean, scaler_coords)

# %%
pred_mean_un = scaler_coords.inverse_transform(pred_mean)
pred_std_un = scaler_coords.inverse_transform(pred_std)
# should we do a 
# radius = np.sqrt((pred_mean_un.transpose()[0] - pred_std_un.transpose()[0])**2 + 
#                  (pred_mean_un.transpose()[1] - pred_std_un.transpose()[1])**2)

# %%
print('mean x coords')
print(pred_mean_un.transpose()[0])

# %%
print('std x coords')
print(pred_std_un.transpose()[0])

# %%
print('mean y coords')
print(pred_mean_un.transpose()[1])

# %%
print('std y coords')
print(pred_std_un.transpose()[1])

# %%



