import os
# import sys 
# sys.path.append('/media/vn/Data/Workspace/leakage_detection_cfrp')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# Libraries
import numpy as np
# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)
import pandas as pd

np.random.seed(0)

def load_data():
    # data = pd.read_csv('../data_acquisition/wing_leakage_data_samples_filt_bad_out.csv')
    # data = data.drop(data[data.quality == 'bad'].index)
    # # data = data[data['MFC9'].notna()]
    # data = data.drop(data[data.sample_number > total_samples].index)
    # data = data.drop(data[data.Comments == 'outside'].index)
    # data = data.drop(data[data.Comments == 'Repeat'].index)
    # data = data.drop(data[data.Comments == 'repeat'].index)
    # data = data.drop(data[data.Comments == 'on port'].index)
    # data = data.drop(data[data.Comments == 'missed'].index)
    # data = data.drop(data[data.Comments == 'Not sure if I will be able to seal the leakage on y axis'].index)
    # data = data.drop(columns=['sample_number', 'total flow rate','Comments','Day'])
    # data = data.drop(columns=['quality'])
    # data = data.rename(columns={'number of leakage':'number_of_leakage'})
    # # data.columns
    # single_leakage = data[data['number_of_leakage'] == 1]
    # single_leakage = single_leakage.drop(columns=['x2', 'y2'])
    # two_leakage = data[data['number_of_leakage'] == 2]

    # single_leakage = single_leakage.drop(columns=['number_of_leakage'])
    # two_leakage = two_leakage.drop(columns=['number_of_leakage'])
    # single_leakage.to_csv("single_leakage.csv")

    data = pd.read_csv('../data_acquisition/clean_data_check2_less.csv', index_col=0)
    data = data.drop(columns=['Comments', 'Day', 'quality'])
    single_leakage = data[data['number_of_leakage'] == 1].drop(columns=['number_of_leakage', 'x2', 'y2'])
    double_leakage = data[data['number_of_leakage'] == 2].drop(columns=['number_of_leakage'])
    return single_leakage, double_leakage

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

def split_xy(data):
    y = data[['x1', 'y1']] 
    x = data.drop(['x1', 'y1'], axis=1)
    return x,y


def data_split(single_leakage):
    x,y = split_xy(single_leakage)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1) # 0.25 x 0.8 = 0.2

    model_data = [X_train, X_test, y_train, y_test, X_val, y_val]
    file_name = ['X_train', 'X_test', 'y_train', 'y_test', 'X_val', 'y_val']

    for data, file_name in zip(model_data, file_name):
        data.to_csv("./model_data/" + file_name + ".csv")


# def mfc_sum_scaler(flow_data):
#     mfc_sum = np.sum(flow_data, axis=1)
#     flow_data = flow_data / mfc_sum[:,None]
#     return flow_data, mfc_sum

def normalize_data(X_train, X_test, X_val, y_train, y_test, y_val):

    scaler_coords = StandardScaler()
    y_train = scaler_coords.fit_transform(y_train)
    y_test = scaler_coords.transform(y_test)
    y_val = scaler_coords.transform(y_val)

    scaler_flows = StandardScaler()
    X_train = scaler_flows.fit_transform(X_train)
    X_test = scaler_flows.transform(X_test)
    X_val = scaler_flows.transform(X_val)

    return X_train, X_test, X_val, y_train, y_test, y_val, scaler_coords, scaler_flows

# def scale_transform_coords(data):
#     data[:, 0:1] = data[:, 0:1] * 2 - 1
#     data[:, 1:2] = data[:, 1:2] * 2 - 1

#     return data

def load_single_leakage_model_data(residual_subtract,
                                   augmentation, 
                                   blind_flip, 
                                   tot_flow, 
                                   res_flow, 
                                   tot_resflow
                                   ):
    X_train = pd.read_csv('./model_data/X_train.csv')
    X_test = pd.read_csv('./model_data/X_test.csv')
    X_val = pd.read_csv('./model_data/X_val.csv')

    y_train = pd.read_csv('./model_data/y_train.csv').to_numpy()
    y_test = pd.read_csv('./model_data/y_test.csv')
    y_val = pd.read_csv('./model_data/y_val.csv')

    if residual_subtract is True:
        X_train, X_test, X_val = reduce_residual(X_train, X_test, X_val)
    
    X_train = X_train.to_numpy()
    

    # add logic for augmentation
    # if augmentation is True:
    #     X_train_flip, y_train_flip = flipped_data(X_train, y_train, blind_flip)
    #     print(X_train_flip.columns, y_train_flip.shape)
    #     print(X_train.columns, y_train.shape)
    #     X_train = np.vstack((X_train, X_train_flip))
    #     y_train = np.vstack((y_train, y_train_flip))

    X_train = pd.DataFrame(X_train, columns=X_test.columns)
    y_train = pd.DataFrame(y_train, columns=y_test.columns)

    if tot_flow is False:
        X_train = X_train.drop(columns=['total flow rate'])
        X_test = X_test.drop(columns=['total flow rate'])
        X_val = X_val.drop(columns=['total flow rate'])

    if res_flow is False:
        for i in range(1,11):
            X_train = X_train.drop(columns=['mfc'+str(i)+'_residual'])
            X_test = X_test.drop(columns=['mfc'+str(i)+'_residual'])
            X_val = X_val.drop(columns=['mfc'+str(i)+'_residual'])
        
    if tot_resflow is False:
        X_train = X_train.drop(columns=['tot_residual_flow'])
        X_test = X_test.drop(columns=['tot_residual_flow'])
        X_val = X_val.drop(columns=['tot_residual_flow'])

    X_train, X_test, X_val, y_train, y_test, y_val, scaler_coords, scaler_flows = normalize_data(X_train, X_test, X_val, y_train, y_test, y_val)

    return X_train, X_test, X_val, y_train, y_test, y_val, scaler_coords, scaler_flows

def reduce_residual(X_train, X_test, X_val):
    # print(X_train.columns)
    for i in range(1,11):
        column_name = str('MFC'+str(i))
        # print(column_name)
        res_column_name = str('mfc'+str(i)+'_residual')
        X_train[column_name] = X_train[column_name] - X_train[res_column_name]
        X_test[column_name] = X_test[column_name] - X_test[res_column_name]
        X_val[column_name] = X_val[column_name] - X_val[res_column_name]
        X_train = X_train.drop(columns=[res_column_name])
        X_test = X_test.drop(columns=[res_column_name])
        X_val = X_val.drop(columns=[res_column_name])

    return X_train, X_test, X_val
