# %%
import tensorflow as tf
from utils.data_preprocess import load_data
from utils.module import model_eval
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from utils.model_evaluation import plot_test_pred

# Sandwiched dropout layer to the hyperparameter tuned model and did the predictions 1000 times without training the model again
# Sandwiched dropout layer to the hyperparameter tuned model and did the predictions 1000 times after training the model again
# During hyperparameter tuning, it is ensured that a dropout layer is there after a dense layer. This model is directly used for uncertainity quanitification

single_leakage, two_leakage = load_data()
# print(single_leakage.columns)
data = single_leakage.drop(columns=['total flow rate', 'mfc6_residual',
       'mfc7_residual', 'mfc8_residual', 'mfc9_residual', 'mfc10_residual',
       'mfc1_residual', 'mfc2_residual', 'mfc3_residual', 'mfc4_residual',
       'mfc5_residual'])

print(data.columns)
print(data.shape)

y = data[['x1', 'y1']]
x = data.drop(['x1', 'y1'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1) 

scaler_coords = StandardScaler()
y_train = scaler_coords.fit_transform(y_train)
y_test = scaler_coords.transform(y_test)
y_val = scaler_coords.transform(y_val)

scaler_flows = StandardScaler()
X_train = scaler_flows.fit_transform(X_train)
X_test = scaler_flows.transform(X_test)
X_val = scaler_flows.transform(X_val)

# %%
dropout_prob = 0.1

model = tf.keras.models.load_model('saved_model/single_leak/single_leakage_model_less')
print(model.summary())
model_evaluate, y_pred = model_eval(model, X_test, y_test, X_train, y_train, X_val, y_val)

stoch_model = tf.keras.Sequential()

for i, layer in enumerate(model.layers):
    stoch_model.add(layer)
    # Add your intermediate layer after each existing layer
    if i == 0:
        continue
    if i == len(model.layers)-1:
        continue
    intermediate_layer = tf.keras.layers.Dropout(dropout_prob)
    stoch_model.add(intermediate_layer)

# Compile the new model
stoch_model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=model.optimizer.lr.numpy()),
                loss="mse",
                metrics='mae')
# Print the summary of the new model
stoch_model.summary()
model_evaluate, y_pred = model_eval(stoch_model, X_test, y_test, X_train, y_train, X_val, y_val)
# # # %%
pred=np.stack([stoch_model(X_test,training=True) 
               for sample in range(1000)])
predictions_list = pred.tolist()
predictions_list_unsc = []
# print(len(predictions_list))
for pred in predictions_list:
    pred = scaler_coords.inverse_transform(pred)
    predictions_list_unsc.append(pred)
predictions__unsc = np.array(predictions_list_unsc)

# print(predictions__unsc.shape)
pred_mean=predictions__unsc.mean(axis=0)
pred_std = predictions__unsc.std(axis=0) 
# print(pred_mean.shape, pred_std.shape)
# # %%
y_test = scaler_coords.inverse_transform(y_test)
plot_test_pred(y_test, pred_mean)
print(pred_std)


# # # %%
# pred_mean_un = scaler_coords.inverse_transform(pred_mean)
# pred_std_un = scaler_coords.inverse_transform(pred_std)
# should we do a 
# radius = np.sqrt((pred_mean_un.transpose()[0] - pred_std_un.transpose()[0])**2 + 
#                  (pred_mean_un.transpose()[1] - pred_std_un.transpose()[1])**2)

# # %%
# print('mean x coords')
# print(pred_mean_un.transpose()[0])

# # # %%
# print('std x coords')
# print(pred_std_un.transpose()[0])

# # # %%
# print('mean y coords')
# print(pred_mean_un.transpose()[1])

# # # %%
# print('std y coords')
# print(pred_std_un.transpose()[1])

# # # %%