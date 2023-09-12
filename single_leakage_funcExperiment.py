from utils.data_preprocess import load_data
from utils.module import hyper_func_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from utils.module import model_eval
from utils.model_evaluation import plot_test_pred
import tensorflow as tf

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
# hyper_model.save('saved_model/single_leak/single_leakage_final_cleansed')

hyper_model = tf.keras.models.load_model('saved_model/single_leak/single_leakage_model_less')
hyper_model.fit(X_train, y_train, epochs=1000, validation_data = (X_val, y_val), shuffle= True)
hyper_model.save('saved_model/single_leak/single_leakage_final_cleansed')

model_evaluate, y_pred = model_eval(hyper_model, X_test, y_test, X_train, y_train, X_val, y_val)
plot_test_pred(y_test, y_pred, scaler_coords)