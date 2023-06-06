import numpy as np
np.random.seed(123)
from utils import *


if __name__ == "__main__":
    #load the input data
    X_train, X_train_augmented,X_test, X_test_augmented,Y_train,Y_train_augmented, Y_test, Y_test_augmented = load_data()
    EPOCH = 25
    #train the standard model
    best_hps,stand_model, tuner = standard_model(X_train,Y_train,EPOCH)
    #train the informed model
    proj_model = projection_model(X_train,Y_train,best_hps,tuner, EPOCH)
    i = 11
    #save the models
    stand_model.save("./submission/final_saved_models/standard_model_"+ str(i))
    proj_model.save("./submission/final_saved_models/projection_model_" + str(i))
    #calculate the prediction accuracy
    graph_accuracy(stand_model, "Standard",i)
    graph_accuracy(proj_model, "Projection",i)
    #calculate the training loss
    train_accuracy(X_train, Y_train,stand_model, "Standard model")
    train_accuracy(X_train, Y_train, proj_model, "Projection model")
    #calculate the test loss
    test_accuracy(X_test, Y_test, stand_model, "Standard model")
    test_accuracy(X_test, Y_test, proj_model, "Projection model")
