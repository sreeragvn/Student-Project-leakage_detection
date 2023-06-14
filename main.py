# %%
from utils.data_preprocess import load_data, split_xy_save, load_single_leakage_model_data
from utils.module import model_eval, hyper_model, model_comparison, linear_regression, numpy_to_tensor, benchmark_linear_model
import itertools
import pandas as pd 
# single_leakage_data_spread(single_leakage)
augmentation = [True, False] # set aug = True to load augmented where the data is flipped across x axis through the centre
blind_flip = [True, False]
residual_subtract = [True, False]
tot_mfc_scaler = [True, False]
experiment_repeat =1

span = 16048
width = 5233
total_samples = 862
EPOCH = 1
factor = 20
rebuild_train = False
test = list(itertools.product(augmentation,residual_subtract,tot_mfc_scaler, blind_flip))
# %%
single_leakage, two_leakage = load_data(total_samples = total_samples)
# variance_plot(single_leakage)
# %%
# data_check(single_leakage) # Average are not really close together. should we be worried ?
if rebuild_train:
    split_xy_save(single_leakage)
    print('data rebuilt')
# %%
model_metric = pd.DataFrame()
model_performance_file = "results/model_performance.csv"
model_metric.to_csv(model_performance_file)
for i in range(experiment_repeat):
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
        model_evaluate, y_pred = linear_regression(X_train, y_train, X_test, y_test, scaler_coords, X_val, y_val)
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
                                                        EPOCH, factor,augmentation,
                                                        residual_subtract,tot_mfc_scaler,blind_flip)
        model_evaluate, y_pred = model_eval(best_model, X_test, y_test, X_train, y_train, X_val, y_val)
        model_metric = model_comparison(model_metric, model_evaluate, augmentation = augmentation, 
                                        residual_subtract = residual_subtract, mfc_sum_scale = tot_mfc_scaler, 
                                        blind_flip = blind_flip, label = "nn_hyper_model"+str(i))
        model_metric.to_csv(model_performance_file, index=False)
# %%