import os
# import sys 
# sys.path.append('/media/vn/Data/Workspace/leakage_detection_cfrp')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
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
from utils.model_evaluation import plot_wing_contour

x_range = np.arange(180, 16048, 250)
y_range = np.arange(100, 5233, 250)
X, Y = np.meshgrid(x_range, y_range)

def data_check(data):
    data_plot(data)
    print(data.isna().any())
    print(data.describe().transpose())
    sns.pairplot(data[data.columns.values], diag_kind='kde')

def get_leakage_close_to_sensor(db_type):
    if db_type == 'xlsx':
        data = pd.read_excel('../data_acquisition/wing_leakage_data_samples_filt.xlsx', index_col=0)
    # print(data.columns)
    else:
        data = pd.read_csv('../data_acquisition/wing_leakage_data_samples_filt_bad_out.csv', index_col=0)
    # data = data.drop(data[data.quality == 'bad'].index)
    # data = data.drop(columns=[])
    data = data.rename(columns={'number of leakage':'number_of_leakage'})
    # data = data.dropna(subset=['x1'])
    data = data.dropna(subset=['MFC6'])
    single_leakage = data[data['number_of_leakage'] == 1]
    double_leakage = data[data['number_of_leakage'] == 2]
    cols_to_remove = ['total flow rate','Comments', 'Day', 'number_of_leakage', 
                      'quality', 'mfc10_residual','mfc9_residual', 'mfc7_residual',	'mfc8_residual',
                      'mfc4_residual', 'mfc5_residual','mfc1_residual', 'mfc2_residual', 'mfc3_residual', 
                      'mfc6_residual']
    single_leakage = single_leakage.drop(columns=cols_to_remove)
    single_leakage = single_leakage.drop(columns=['x2', 'y2'])
    flows = single_leakage.drop(columns=['x1', 'y1'])
    double_leakage = double_leakage.drop(columns=cols_to_remove)
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

    return data, flows, single_leakage, double_leakage, mfc1, mfc2, mfc3, mfc4, mfc5, mfc6, mfc7, mfc8, mfc9, mfc10

def get_index_anomaly(mfc, coord, level, less):
    if not less:
        return mfc.loc[(mfc[coord] >= level)]
    else:
        return mfc.loc[(mfc[coord] <= level)]
    
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
    fig.tight_layout()
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
def meshgrid_to_coords(row):
    i = row[0]
    j = row[1]
    x = X[-j + 10, i+31]
    y = Y[-j + 10, i+31]
    print(x, y)
    # data['tranformed'] = data.apply(meshgrid_to_coords, axis=1)
    # data.to_csv("../data_new.csv", index=False)

    return str(x)+' , '+ str(y)

    # plt.plot([0, 7930], [0, 0], 'k')
    # plt.plot([7930, 16048], [0, 1149], 'k')
    # plt.plot([16048, 16048], [1149, 4386], 'k')
    # plt.plot([16048, 7843], [4386, 5233], 'k')
    # plt.plot([7843, 2493], [5233, 5233], 'k')
    # plt.plot([2493, 0], [5233, 0], 'k')
    # plt.xlim([-1000, 17000])
    # plt.ylim([-1000, 6000])
    # plt.xticks([])
    # plt.yticks([])
    # plt.gca().set_aspect('equal')
    