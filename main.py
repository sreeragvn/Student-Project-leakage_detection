# %%
from utils.data_preprocess import load_data, split_xy_save, load_single_leakage_model_data
from utils.module import model_eval, hyper_model, model_comparison, linear_regression, numpy_to_tensor, benchmark_linear_model
import itertools
import pandas as pd 
import yaml
import tensorflow as tf

with open("config.yml", "r") as ymlfile:
    cfg = yaml.full_load(ymlfile)

test = list(itertools.product(cfg['experiment']['augmentation'], 
                              cfg['experiment']['residual_subtract'],
                              cfg['experiment']['tot_mfc_scaler'],
                              cfg['experiment']['blind_flip']))
# %%

# %%
if cfg['experiment']['rebuild_train']:
    single_leakage, two_leakage = load_data(total_samples = cfg['experiment']['total_samples'])
    split_xy_save(single_leakage)
    print('data rebuilt')
# %%
model_metric = pd.DataFrame()
model_performance_file = "results/model_performance.csv"
model_metric.to_csv(model_performance_file)
for i in range(cfg['experiment']['experiment_repeat']):
    for augmentation, residual_subtract, tot_mfc_scaler, blind_flip in test:
        if augmentation == False and blind_flip == True:
            continue
        if augmentation != False or residual_subtract != False or tot_mfc_scaler != False or blind_flip != False:
            continue
        model_metric = pd.read_csv(model_performance_file, index_col=None)
        X_train, X_test, X_val, y_train, y_test, y_val, scaler_coords, scaler_flows = load_single_leakage_model_data(augmentation,
                                                                                                             residual_subtract,
                                                                                                              tot_mfc_scaler, 
                                                                                                              blind_flip)
        # print(X_train.shape)
        # print(X_test.shape)
        # print(X_val.shape)
        # print(y_train.shape)
        # print(y_test.shape)
        # print(y_val.shape)
        model_evaluate, y_pred = linear_regression(X_train, y_train, X_test, y_test, X_val, y_val)
        model_metric = model_comparison(model_metric, model_evaluate, augmentation = augmentation, 
                                        residual_subtract = residual_subtract, mfc_sum_scale = tot_mfc_scaler, 
                                        blind_flip = blind_flip, label = "linear_regression")
    # %%
        X_train, X_test, X_val, y_train, y_test, y_val = numpy_to_tensor(X_train, X_test, X_val, y_train, y_test, y_val)

        linear_model, history = benchmark_linear_model(X_train, y_train, X_val, y_val)
        model_evaluate, y_pred = model_eval(linear_model, X_test, y_test, X_train, y_train, X_val, y_val)
        model_metric = model_comparison(model_metric, model_evaluate, augmentation = augmentation, 
                                        residual_subtract = residual_subtract, mfc_sum_scale = tot_mfc_scaler, 
                                        blind_flip = blind_flip, label = "nn_linear_regression_"+str(i))

    # %%
    # Hyper parameter tuning
        best_hps, best_model, tuner, history = hyper_model(X_train,y_train, X_val, y_val,
                                                        cfg['experiment']['EPOCH'], cfg['experiment']['factor'],augmentation,
                                                        residual_subtract,tot_mfc_scaler,blind_flip)
        model_evaluate, y_pred = model_eval(best_model, X_test, y_test, X_train, y_train, X_val, y_val)
        model_metric = model_comparison(model_metric, model_evaluate, augmentation = augmentation, 
                                        residual_subtract = residual_subtract, mfc_sum_scale = tot_mfc_scaler, 
                                        blind_flip = blind_flip, label = "nn_hyper_model"+str(i))
        model_metric.to_csv(model_performance_file, index=False)
    break
# %%
best_model.summary()
best_model.save('saved_model/single_leakage_model')
