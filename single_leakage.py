# %%
from utils.data_preprocess import load_single_leakage_model_data
from utils.module import model_eval, hyper_model, model_comparison, linear_regression, numpy_to_tensor, benchmark_linear_model, get_activation_functions
import itertools
import pandas as pd 
import yaml
import numpy as np

with open("config.yml", "r") as ymlfile:
    cfg = yaml.full_load(ymlfile)

experiment = list(itertools.product(cfg['experiment']['augmentation'], 
                              cfg['experiment']['residual_subtract'],
                              cfg['experiment']['blind_flip'],
                              cfg['experiment']['tot_flow'],
                              cfg['experiment']['res_flow'],
                              cfg['experiment']['tot_resflow'],
                              ))

# %%
model_metric = pd.DataFrame()
model_performance_file = "results/model_performance.csv"
model_metric.to_csv(model_performance_file)
for i in range(cfg['experiment']['experiment_repeat']):
    for augmentation, residual_subtract, blind_flip, tot_flow, res_flow, tot_resflow in experiment:
        if augmentation == False and blind_flip == True:
            continue
        if residual_subtract == True and res_flow == False:
            continue
        model_metric = pd.read_csv(model_performance_file, index_col=None)
        X_train, X_test, X_val, y_train, y_test, y_val, scaler_coords, scaler_flows  = load_single_leakage_model_data(residual_subtract, augmentation, 
                                                                                                    blind_flip, tot_flow, 
                                                                                                    res_flow, tot_resflow)
        # print(X_train.shape)
        # print(X_test.shape)
        # print(X_val.shape)
        # print(y_train.shape)
        # print(y_test.shape)
        # print(y_val.shape)

        model_evaluate, y_pred = linear_regression(X_train, y_train, X_test, y_test, X_val, y_val)
        model_metric = model_comparison(model_metric, model_evaluate, augmentation = augmentation, residual_subtract = residual_subtract,
                                        blind_flip = blind_flip, tot_flow = tot_flow, res_flow = res_flow, tot_resflow = tot_resflow,
                                        label = "linear_regression",)
    # %%
        X_train, X_test, X_val, y_train, y_test, y_val = numpy_to_tensor(X_train, X_test, X_val, y_train, y_test, y_val)

        linear_model, history = benchmark_linear_model(X_train, y_train, X_val, y_val)
        model_evaluate, y_pred = model_eval(linear_model, X_test, y_test, X_train, y_train, X_val, y_val)
        model_metric = model_comparison(model_metric, model_evaluate, augmentation = augmentation, residual_subtract = residual_subtract,
                                        blind_flip = blind_flip, tot_flow = tot_flow, res_flow = res_flow, tot_resflow = tot_resflow,
                                        label = "nn_linear_regression_"+str(i))

    # %%
    # Hyper parameter tuning
        best_hps, best_model, tuner, history = hyper_model(X_train,y_train, X_val, y_val,
                                                        cfg['experiment']['EPOCH'], cfg['experiment']['factor'], augmentation, 
                                                        residual_subtract, blind_flip, tot_flow, res_flow, tot_resflow)
        model_evaluate, y_pred = model_eval(best_model, X_test, y_test, X_train, y_train, X_val, y_val)
        model_metric = model_comparison(model_metric, model_evaluate = model_evaluate, augmentation = augmentation, 
                                        residual_subtract= residual_subtract, blind_flip = blind_flip, 
                                        tot_flow = tot_flow, res_flow = res_flow, tot_resflow = tot_resflow,
                                        label = "nn_hyper_model"+str(i))
        model_metric.to_csv(model_performance_file, index=False)
# %%
# get_activation_functions(best_model)
best_model.summary()
best_model.save('saved_model/single_leakage_model_new')