import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# generate leakage grid
x_range = np.arange(180, 16048, 250)
y_range = np.arange(100, 5233, 250)
X, Y = np.meshgrid(x_range, y_range)


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
    

def plot_target(sample_number, x1, y1, i1, j1, x2=None, y2=None, i2=None, j2=None):
    plt.figure(figsize=(40, 20))
    
    plt.title(f'Sample Number {sample_number}', fontsize=20)
    
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
    
    plt.scatter(x1, y1, color='tab:red', s=200)
    
    print(X.shape)
    
    if x2:
        plt.scatter(x2, y2, color='tab:red', s=200)

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
    plt.text(180, 5800, f'(x1, y1) = ({x1}, {y1}) = ({j1-31}, {-i1+10})', fontsize=20)
    if x2:
        plt.text(180 + 32 * 250, 5800, f'(x2, y2) = ({x2}, {y2}) = ({j2-31}, {-i2+10})', fontsize=20)

    # invert y axis
    plt.gca().invert_yaxis()

    plt.show()
    
    
def generate_single_target(sample_number):
    np.random.seed(0)
    for _ in range(sample_number + 1):
        i = np.random.randint(low=0, high=21)
        j = np.random.randint(low=0, high=64)
    return X[i, j], Y[i, j], i, j


def generate_target(sample_number):
    if np.mod(sample_number, 2) == 0:
        x1, y1, i1, j1 = generate_single_target(sample_number // 2)
        x2, y2, i2, j2 = None, None, None, None
    else:
        x1, y1, i1, j1 = generate_single_target((sample_number - 1) // 2)
        x2, y2, i2, j2 = generate_single_target((sample_number - 1) // 2 + 1)
    return x1, y1, i1, j1, x2, y2, i2, j2


def generate_csv(number_of_samples):
    df = pd.DataFrame(columns=['sample_number', 'x1', 'y1', 'x2', 'y2'])
    for sample_number in range(number_of_samples):
        x1, y1, _, _, x2, y2, _, _ = generate_target(sample_number)
        df = pd.concat([df, pd.DataFrame(np.array([[sample_number, x1, y1, x2, y2]]), columns=df.columns)])
    df.to_csv('wing_leakage_data_samples_v2.csv', index=False)