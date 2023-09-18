from utils.data_preprocess import load_data
from utils.module import hyper_func_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from utils.module import model_eval
from utils.model_evaluation import plot_test_pred
import tensorflow as tf
import numpy as np

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

model_data = [X_train, X_test, y_train, y_test, X_val, y_val]
file_name = ['X_train', 'X_test', 'y_train', 'y_test', 'X_val', 'y_val']

for data, file_name in zip(model_data, file_name):
    data = pd.DataFrame(data)
    data.to_csv("./model_data_final_cleansed/" + file_name + ".csv")

scaler_coords = StandardScaler()
y_train = scaler_coords.fit_transform(y_train)
y_test = scaler_coords.transform(y_test)
y_val = scaler_coords.transform(y_val)

scaler_flows = StandardScaler()
X_train = scaler_flows.fit_transform(X_train)
X_test = scaler_flows.transform(X_test)
X_val = scaler_flows.transform(X_val)

# hyper_model = hyper_func_model(X_train, y_train, X_val, y_val, epochs=1000, input_num=10, factor= 2)
# hyper_model.summary()
# # also named single_leakage_model_less
# hyper_model.save('saved_model/single_leak/single_leakage_model_less')

hyper_model = tf.keras.models.load_model('saved_model/single_leak/single_leakage_model_less')
hyper_model.summary()
# from tensorflow.keras.callbacks import ModelCheckpoint
# checkpoint_callback = ModelCheckpoint(
#     filepath='best_single_model_weights.h5',  # Filepath to save the weights
#     monitor='val_mae',               # Metric to monitor for saving
#     save_best_only=True,              # Save only the best model
#     save_weights_only=True,           # Save only the weights (not the full model)
#     mode='min',                       # Mode to minimize the monitored metric
#     verbose=1                          # Verbosity level (optional)
# )

# history = hyper_model.fit(X_train, y_train, epochs=1000, 
#                           validation_data = (X_val, y_val), 
#                           shuffle= True, 
#                           callbacks=[checkpoint_callback]
#                           )
# hyper_model.save('saved_model/single_leak/single_leakage_final_cleansed')

model_evaluate, y_pred = model_eval(hyper_model, X_test, y_test, X_train, y_train, X_val, y_val)
# plot_test_pred(y_test, y_pred, scaler_coords)
y_pred = scaler_coords.inverse_transform(y_pred)
y_test = scaler_coords.inverse_transform(y_test)
print(y_pred.shape, y_test.shape)
diff = y_pred - y_test
distances = np.sqrt(np.sum(diff ** 2, axis=-1))/10
# print(distances)
# print(len(distances))

y_mean = [distances.mean(axis=0)]*len(distances)
ind = np.argpartition(distances, -40)[-40:]
top5 = distances[ind]
average_of_75 = [top5.mean(axis=0)]*len(distances)
# average_of_75 = [30]*len(distances)
print(average_of_75, y_mean)


import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.scatter(range(len(distances)),distances, label = 'L2 distance', color = 'k')
plt.plot(range(len(distances)),y_mean, label='Mean', linestyle='--', color = 'k')
plt.plot(range(len(distances)),average_of_75, label='75% Data', linestyle='-.', color = 'k')
plt.legend(loc="upper left")
# plt.title("Distance between the Model predictions and True value", fontsize = 15)
plt.xlabel('sample no')
plt.ylabel('L2 distance(cm)')
plt.savefig('./results/single_leakage_distance_plot.png')
plt.show()