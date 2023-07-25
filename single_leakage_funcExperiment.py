from utils.data_preprocess import load_data
from utils.module import hyper_func_model
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

with open("config_multi.yml", "r") as ymlfile:
    cfg = yaml.full_load(ymlfile)


single_leakage, two_leakage = load_data(total_samples = cfg['experiment']['total_samples'])
# Not sure if 0 is good enough or try generating a random number

data = single_leakage.drop(columns=['mfc6_residual',
       'mfc7_residual', 'mfc8_residual', 'mfc9_residual', 'mfc10_residual',
       'mfc1_residual', 'mfc2_residual', 'mfc3_residual', 'mfc4_residual',
       'mfc5_residual', 'tot_residual_flow', 
       'total flow rate'
       ])

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

hyper_model = hyper_func_model(X_train, y_train, X_val, y_val, epochs=1000, input_num=10, factor= 2)
hyper_model.summary()
hyper_model.save('saved_model/single_leakage_model_func')
