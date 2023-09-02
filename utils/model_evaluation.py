# import sys 
# sys.path.append('/media/vn/Data/Workspace/leakage_detection_cfrp')
import warnings
warnings.filterwarnings("ignore")
from typing import Optional

# Libraries
import numpy as np
# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)
import matplotlib.pyplot as plt

# generate leakage grid
x_range = np.arange(180, 16048, 250)
y_range = np.arange(100, 5233, 250)
X, Y = np.meshgrid(x_range, y_range)

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

def plot_test_pred(test, pred, scaler_coords : Optional[object] = None):
    # pred = pred * test_flow_sum[:,None]
    # test = test * test_flow_sum[:,None]

    # pred[:, 0:1] = (pred[:, 0:1] + 1) / 2
    # pred[:, 1:2] = (pred[:, 1:2] + 1) / 2
    # test[:, 0:1] = (test[:, 0:1] + 1) / 2
    # test[:, 1:2] = (test[:, 1:2] + 1) / 2

    if scaler_coords is not None:
        print("scaled")
        pred = scaler_coords.inverse_transform(pred)
        test = scaler_coords.inverse_transform(test)
    plt.figure(figsize=(40, 20))
    
    # plt.title(f'Sample Number {sample_number}', fontsize=20)
    
    # plot sensor positions
    sensors = np.array([[2426, 70], [5480, 70], [8661, 191], [11676, 584], [13976, 917], [2603, 5163], [5723, 5163], [8417, 5103], [11646, 4740], [14641, 4391]])
    for i in range(len(sensors)):
        plt.scatter(sensors[i, 0], sensors[i, 1], color='tab:green', s=300)
        if i < 5:
            plt.text(sensors[i, 0], sensors[i, 1] - 200, 'MFC'+str(i+1), fontsize='xx-large')
        else:
            plt.text(sensors[i, 0], sensors[i, 1] + 350, 'MFC'+str(i+1), fontsize='xx-large')

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

def plot_wing_contour():
    plt.plot([0, 7930], [0, 0], 'k')
    plt.plot([7930, 16048], [0, 1149], 'k')
    plt.plot([16048, 16048], [1149, 4386], 'k')
    plt.plot([16048, 7843], [4386, 5233], 'k')
    plt.plot([7843, 2493], [5233, 5233], 'k')
    plt.plot([2493, 0], [5233, 0], 'k')
    plt.xlim([-1000, 17000])
    plt.ylim([-1000, 6000])
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_aspect('equal')