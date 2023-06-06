import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
import time
import sys
from scipy.spatial import distance
import matplotlib.pyplot as plt
#tf.random.set_seed(123)


# auxiliary function to rotate samples
def rotate90(df):
    df_90 = df.copy()
    df_90['mfc1'] = df['mfc4']
    df_90['mfc2'] = df['mfc1']
    df_90['mfc3'] = df['mfc2']
    df_90['mfc4'] = df['mfc3']
    for i in [1, 2, 3]:
        df_90[f'x{i}'] = 1 - df[f'y{i}']
        df_90[f'y{i}'] = df[f'x{i}']
    df_90['angle'] = df['angle'] + 90
    return df_90

# auxiliary function to flip samples
def flip(df):
    df_flip = df.copy()
    df_flip['mfc1'] = df['mfc2']
    df_flip['mfc2'] = df['mfc1']
    df_flip['mfc3'] = df['mfc4']
    df_flip['mfc4'] = df['mfc3']
    for i in [1, 2, 3]:
        df_flip[f'x{i}'] = 1 - df[f'x{i}']
        df_flip[f'y{i}'] = df[f'y{i}']
    df_flip['flipped'] = True
    return df_flip
#load the input csv file and read the input data
def load_data():
    df = pd.read_csv('data.csv', index_col=0)

    # augment data
    df['angle'] = 0
    df = df.append(rotate90(df))
    df = df.append(rotate90(df[(df['angle'] == 90)]))
    df = df.append(rotate90(df[(df['angle'] == 180)]))
    df['flipped'] = False
    df = df.append(flip(df))

    # keep only 1-leakage samples (only these are considered in this paper)
    df = df[df['n'] == 1]

    # normalize total flow so that x_1 + x_2 + x_3 + x_4 = 1
    df['mfcsum'] = df['mfc1'] + df['mfc2'] + df['mfc3'] + df['mfc4']
    df[['mfc1', 'mfc2', 'mfc3', 'mfc4']] = df[['mfc1', 'mfc2', 'mfc3', 'mfc4']].div(df['mfcsum'], axis=0)

    # remove obsolete columns
    df.drop(['s1', 'x2', 'y2', 's2', 'x3', 'y3', 's3', 'n', 'mfcsum'], axis=1, inplace=True)

    # apply coordinate transformation to get (y_1, y_2) in [-1, 1] x [-1, 1]
    df['x1'] = df['x1'].map(lambda z: 2 * z - 1)
    df['y1'] = df['y1'].map(lambda z: -2 * z + 1)

    # get training and test data
    X_train = df.loc[(df['split'] == 'train') & (df['angle'] == 0) & (df['flipped'] == False), ['mfc1', 'mfc2', 'mfc3',
                                                                                                'mfc4']].values.astype(
        np.float32)
    X_train_augmented = df.loc[(df['split'] == 'train'), ['mfc1', 'mfc2', 'mfc3', 'mfc4']].values.astype(np.float32)
    X_test = df.loc[(df['split'] == 'test') & (df['angle'] == 0) & (df['flipped'] == False), ['mfc1', 'mfc2', 'mfc3',
                                                                                              'mfc4']].values.astype(
        np.float32)
    X_test_augmented = df.loc[(df['split'] == 'test'), ['mfc1', 'mfc2', 'mfc3', 'mfc4']].values.astype(np.float32)
    Y_train = df.loc[
        (df['split'] == 'train') & (df['angle'] == 0) & (df['flipped'] == False), ['x1', 'y1']].values.astype(
        np.float32)
    Y_train_augmented = df.loc[(df['split'] == 'train'), ['x1', 'y1']].values.astype(np.float32)
    Y_test = df.loc[
        (df['split'] == 'test') & (df['angle'] == 0) & (df['flipped'] == False), ['x1', 'y1']].values.astype(np.float32)
    Y_test_augmented = df.loc[(df['split'] == 'test'), ['x1', 'y1']].values.astype(np.float32)

    return  X_train, X_train_augmented,X_test, X_test_augmented,Y_train,Y_train_augmented, Y_test, Y_test_augmented

#calculate the training loss
def train_accuracy(X_train, Y_train,model, name):
    train_loss = model.evaluate(X_train, Y_train)
    print(name + " train mse loss: {:.4f}".format(train_loss))

#calculate the test loss
def test_accuracy(X_test, Y_test,model, name):
    test_loss = model.evaluate(X_test, Y_test)
    print(name + " test mse loss: {:.4f}".format(test_loss))


#Define the model using model builder from keras tuner
def model_builder(hp):
  model = keras.Sequential()

  # Choose an optimal value between 32-512
  for i in range(1, hp.Int("num_layers", 4, 6)):
      model.add(
          keras.layers.Dense(
              units=hp.Int("units_" + str(i), min_value=32, max_value=512, step=32),
              activation="relu")
      )
  model.add(keras.layers.Dense(units=2, activation= "linear"))

  # Tune the learning rate for the optimizer
  # Choose an optimal value from 0.01, 0.001, or 0.0001
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

  model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss="mse")

  return model

#search for the best hyperparameters and train the standard model with original training data
def standard_model(X_train,Y_train, EPOCH):
    tuner = kt.Hyperband(model_builder,
                         objective='val_loss',
                         max_epochs=10,
                         factor=3,
                         directory='my_dir',
                         project_name='intro_to_kt')
    tuner.search_space_summary()
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    tuner.search(X_train, Y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])
    #tuner.search(X_train, Y_train, epochs=50, validation_data=(X_test,Y_test), callbacks=[stop_early])
    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
    The hyperparameter search is complete. The optimal learning rate for the optimizer
    is {best_hps.get('learning_rate')}.
    """)

    model = tuner.hypermodel.build(best_hps)
    history = model.fit(X_train, Y_train, epochs=EPOCH,shuffle= True)


    return best_hps, model, tuner


#to calculate the prediction accuracy according to validation metric
def graph_accuracy(model,name,i):
    tmp = np.load('test1.npy')
    idx_1 = (tmp[:, 0] > tmp[:, 1]) & (tmp[:, 0] > tmp[:, 2]) & (tmp[:, 0] > tmp[:, 3])
    idx_2 = (tmp[:, 1] > tmp[:, 0]) & (tmp[:, 1] > tmp[:, 2]) & (tmp[:, 1] > tmp[:, 3])
    idx_3 = (tmp[:, 2] > tmp[:, 0]) & (tmp[:, 2] > tmp[:, 1]) & (tmp[:, 2] > tmp[:, 3])
    idx_4 = (tmp[:, 3] > tmp[:, 0]) & (tmp[:, 3] > tmp[:, 1]) & (tmp[:, 3] > tmp[:, 2])
    data_1 = tmp[idx_1, :]
    data_2 = tmp[idx_2, :]
    data_3 = tmp[idx_3, :]
    data_4 = tmp[idx_4, :]

    pred_standata_1 = model(data_1)
    pred_standata_2 = model(data_2)
    pred_standata_3 = model(data_3)
    pred_standata_4 = model(data_4)
    plt.figure(figsize=(2, 2))
    plt.scatter(pred_standata_1[:, 0], pred_standata_1[:, 1], alpha=0.03)
    plt.scatter(pred_standata_2[:, 0], pred_standata_2[:, 1], alpha=0.03)
    plt.scatter(pred_standata_3[:, 0], pred_standata_3[:, 1], color='lime', alpha=0.03)
    plt.scatter(pred_standata_4[:, 0], pred_standata_4[:, 1], color='silver', alpha=0.03)
    plt.scatter([-.9, -.9, .9, .9], [-.9, .9, .9, -.9], s=50, facecolor='black', linewidth=0.0)
    plt.text(-.75, .8, 'MFC1', fontsize=8)
    plt.text(.42, .8, 'MFC2', fontsize=8)
    plt.text(.42, -.89, 'MFC3', fontsize=8)
    plt.text(-.75, -.89, 'MFC4', fontsize=8)
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.grid(True)
    plt.xticks([0])
    plt.yticks([0])
    plt.gca().set_xticklabels([''])

    plt.gca().set_yticklabels([''])
    plt.gca().set_aspect('equal', 'box')
    plt.savefig("./submission/final_saved_dist/" + name + '_dist_'+ str(i), pad_inches=0.01, bbox_inches='tight', dpi=200)

    val_cor1 = val_metric(pred_standata_1, -1, 0, 0, 1)
    val_cor2 = val_metric(pred_standata_2, 0, 1, 0, 1)
    val_cor3 = val_metric(pred_standata_3, 0, 1, -1, 0)
    val_cor4 = val_metric(pred_standata_4, -1, 0, -1, 0)
    sum_cor = val_cor1 + val_cor2 + val_cor3 + val_cor4
    accuracy = (sum_cor / len(tmp)) * 100
    print(name + "_Accuracy=", accuracy)

#define the validation metric
def val_metric(data,x_min,x_max,y_min,y_max):
    correct_pred=0
    for i in data:
        if ((x_min <= i[0] <= x_max) & (y_min <= i[1] <= y_max)):
            correct_pred = correct_pred + 1
    return correct_pred

#loss function
def mse(Y_true,Y_pred):
    return tf.reduce_mean(tf.square(Y_true-Y_pred))


#define the train step for informed model
def step(X, y, model, opt):
    with tf.GradientTape() as tape:
        pred=model(X)
        loss=mse(y,pred)
    grads=tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))

#predict the values using the trained informed model
def prediction(X,model):
    idx_1 = (X[:, 0] > X[:, 1]) & (X[:, 0] > X[:, 2]) & (X[:, 0] > X[:, 3])
    idx_2 = (X[:, 1] > X[:, 0]) & (X[:, 1] > X[:, 2]) & (X[:, 1] > X[:, 3])
    idx_3 = (X[:, 2] > X[:, 0]) & (X[:, 2] > X[:, 1]) & (X[:, 2] > X[:, 3])
    idx_4 = (X[:, 3] > X[:, 0]) & (X[:, 3] > X[:, 1]) & (X[:, 3] > X[:, 2])
    X_1 = X[idx_1, :]
    X_2 = X[idx_2, :]
    X_3 = X[idx_3, :]
    X_4 = X[idx_4, :]
    P_1 = model(X_1).numpy()
    P_2 = model(X_2).numpy()
    P_3 = model(X_3).numpy()
    P_4 = model(X_4).numpy()
    index_1 = (P_1[:, 0] > -1) & (P_1[:, 0] < 0) & (P_1[:, 1] < 1) & (P_1[:, 1] > 0)
    index_2 = (P_2[:, 0] > 0) & (P_2[:, 0] < 1) & (P_2[:, 1] < 1) & (P_2[:, 1] > 0)
    index_3 = (P_3[:, 0] > 0) & (P_3[:, 0] < 1) & (P_3[:, 1] < 0) & (P_3[:, 1] > -1)
    index_4 = (P_4[:, 0] > -1) & (P_4[:, 0] < 0) & (P_4[:, 1] < 0) & (P_4[:, 1] > -1)
    Yproj_1 = P_1[np.invert(index_1), :]
    Yproj_2 = P_2[np.invert(index_2), :]
    Yproj_3 = P_3[np.invert(index_3), :]
    Yproj_4 = P_4[np.invert(index_4), :]
    Xproj_1 = X_1[np.invert(index_1), :]
    Xproj_2 = X_2[np.invert(index_2), :]
    Xproj_3 = X_3[np.invert(index_3), :]
    Xproj_4 = X_4[np.invert(index_4), :]
    Xproj = np.concatenate([Xproj_1,Xproj_2,Xproj_3,Xproj_4],axis=0)
    return Yproj_1, Yproj_2, Yproj_3, Yproj_4, Xproj

#projection algorithm used while training the informed model
def point_on_line(a, b, p):
    ap = p - a
    ab = b - a
    t = np.dot(ap, ab) / np.dot(ab, ab)
    # if you need the the closest point belonging to the segment
    t = max(0, min(1, t))
    result = a + t * ab
    return result

A= np.array([0,0])
B= np.array([1,0])
C= np.array([1,1])
D= np.array([0,1])
E= np.array([-1,1])
F= np.array([-1,0])
G= np.array([-1,-1])
H= np.array([0,-1])
I= np.array([1,-1])


#using the projection algorithm to project on to the right quadrants
def projection(P, point1, point2, point3, point4):
    P_proj1 = []
    P_proj2 = []
    P_proj3 = []
    P_proj4 = []
    dist1 = []
    dist2 = []
    dist3 = []
    dist4 = []
    for i in range(P.shape[0]):

        P_proj1.append(point_on_line(point1,point2,P[i]))
        P_proj2.append(point_on_line(point2, point3, P[i]))
        P_proj3.append(point_on_line(point3, point4, P[i]))
        P_proj4.append(point_on_line(point4, point1, P[i]))
        dist1.append(distance.euclidean(P[i],P_proj1[i]))
        dist2.append(distance.euclidean(P[i], P_proj2[i]))
        dist3.append(distance.euclidean(P[i], P_proj3[i]))
        dist4.append(distance.euclidean(P[i], P_proj4[i]))
    distance_ = np.column_stack([dist1, dist2, dist3, dist4])
    minInRows = np.argmin(distance_[:], axis=1)
    Proj_final =[]
    for i in range(len(minInRows)):
        if minInRows[i] == 0:
            Proj_final.append(P_proj1[i])
        if minInRows[i] == 1:
            Proj_final.append(P_proj2[i])
        if minInRows[i] == 2:
            Proj_final.append(P_proj3[i])
        if minInRows[i] == 3:
            Proj_final.append(P_proj4[i])
    return Proj_final

#training the informed model with training data and ramdom input data from uniform distribution
def projection_model(X_train,Y_train,best_hps,tuner, EPOCH):
    EPOCHS = EPOCH
    BS = 32
    BS2= 256
    INIT_LR = best_hps.get('learning_rate')
    LR = best_hps.get('learning_rate')
    opt1 = keras.optimizers.Adam(learning_rate=INIT_LR)
    opt2 = keras.optimizers.Adam(learning_rate=LR)
    numUpdates = int(X_train.shape[0] / BS)

    model1 = tuner.hypermodel.build(best_hps)
    #random input data from uniform distribution
    x_data = uniform_dist()
    for epoch in range(0, EPOCHS):
        print("[INFO] starting epoch {}/{}...".format(epoch + 1, EPOCHS), end="")
        sys.stdout.flush()
        epochStart = time.time()
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)

        X_train = X_train[indices]
        Y_train = Y_train[indices]
        for i in range(0, numUpdates):
            start = i * BS
            end = start + BS
            start2 = i * BS2
            end2 = start2 + BS2
            #first training step
            step(X_train[start:end], Y_train[start:end], model=model1, opt=opt1)
            X = x_data[start2:end2]

            P_1, P_2, P_3, P_4, X_proj = prediction(X, model1)
            Proj_final_1 = []
            Proj_final_2 = []
            Proj_final_3 = []
            Proj_final_4 = []
            Y_proj = []
            if P_1.shape[0] > 0:
                Proj_final_1 = projection(P_1, A, D, E, F)
            if P_2.shape[0] > 0:
                Proj_final_2 = projection(P_2, A, B, C, D)
            if P_3.shape[0] > 0:
                Proj_final_3 = projection(P_3, A, H, I, B)
            if P_4.shape[0] > 0:
                Proj_final_4 = projection(P_4, A, F, G, H)
            if len(Proj_final_1) > 0:
                for i in range(len(Proj_final_1)):
                    Y_proj.append(Proj_final_1[i])
            if len(Proj_final_2) > 0:
                for i in range(len(Proj_final_2)):
                    Y_proj.append(Proj_final_2[i])
            if len(Proj_final_3) > 0:
                for i in range(len(Proj_final_3)):
                    Y_proj.append(Proj_final_3[i])
            if len(Proj_final_4) > 0:
                for i in range(len(Proj_final_4)):
                    Y_proj.append(Proj_final_4[i])
            Y_proj = np.asarray(Y_proj)

            if len(X_proj > 0):
                #second training step
                step(X_proj, Y_proj, model=model1, opt=opt2)
        epochEnd = time.time()
        elapsed = (epochEnd - epochStart) / 60.0
        print("took {:.4} minutes".format(elapsed))
    return model1

#Generate random input data from an uniform distribution
def uniform_dist():
    tmp = np.random.uniform(low=0.0, high=1.0, size=(1000000, 4))
    idx_sum = np.sum(tmp, axis=1) <= 1
    tmp = tmp[idx_sum, :]
    idx_1 = (tmp[:, 0] > tmp[:, 1]) & (tmp[:, 0] > tmp[:, 2]) & (tmp[:, 0] > tmp[:, 3])
    idx_2 = (tmp[:, 1] > tmp[:, 0]) & (tmp[:, 1] > tmp[:, 2]) & (tmp[:, 1] > tmp[:, 3])
    idx_3 = (tmp[:, 2] > tmp[:, 0]) & (tmp[:, 2] > tmp[:, 1]) & (tmp[:, 2] > tmp[:, 3])
    idx_4 = (tmp[:, 3] > tmp[:, 0]) & (tmp[:, 3] > tmp[:, 1]) & (tmp[:, 3] > tmp[:, 2])
    data_1 = tmp[idx_1, :]
    data_2 = tmp[idx_2, :]
    data_3 = tmp[idx_3, :]
    data_4 = tmp[idx_4, :]
    x_data = np.concatenate([data_1, data_2, data_3, data_4], axis=0)
    indices = np.arange(x_data.shape[0])
    np.random.shuffle(indices)

    x_data = x_data[indices]
    return x_data