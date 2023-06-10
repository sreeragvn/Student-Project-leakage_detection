# %%
from utils import *
# single_leakage_data_spread(single_leakage)
augmentation = [True, False] # set aug = True to load augmented where the data is flipped across x axis through the centre
blind_flip = [True, False]
residual_subtract = [True, False]
tot_mfc_scaler = [True, False]
experiment_repeat = 5

span = 16048
width = 5233
total_samples = 862
EPOCH = 100
factor = 2
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
model_metric.to_csv("model_performance.csv")
for augmentation, residual_subtract, tot_mfc_scaler, blind_flip in test:
    for i in range(experiment_repeat):
        model_metric = pd.read_csv("model_performance.csv", index_col=None)
        X_train, X_test, X_val, y_train, y_test, y_val, scaler_coords, scaler_flows = load_single_leakage_model_data(str(augmentation),str(residual_subtract), str(tot_mfc_scaler), str(blind_flip))
        model_evaluate = scikit_linear_regression(X_train, y_train, X_test, y_test, scaler_coords, X_val, y_val)
        model_metric = model_comparison(model_metric, model_evaluate, augmentation = str(augmentation),
                                        residual_subtract = str(residual_subtract), mfc_sum_scaler = str(tot_mfc_scaler), 
                                        blind_flip = str(blind_flip), label = "linear_regression")
    # %%

        X_train = tf.convert_to_tensor(X_train, np.float32)
        X_test = tf.convert_to_tensor(X_test, np.float32)
        X_val = tf.convert_to_tensor(X_val, np.float32)
        y_train = tf.convert_to_tensor(y_train, np.float32)
        y_test = tf.convert_to_tensor(y_test, np.float32)
        y_val = tf.convert_to_tensor(y_val, np.float32)

        linear_model, history = benchmark_linear_model(X_train, y_train, X_val, y_val)
        model_evaluate = model_eval(linear_model, X_test, y_test, scaler_coords, X_train, y_train, X_val, y_val, history)
        model_metric = model_comparison(model_metric, model_evaluate, augmentation = str(augmentation), 
                                        residual_subtract = str(residual_subtract), mfc_sum_scaler = str(tot_mfc_scaler), 
                                        blind_flip = str(blind_flip), label = "nn_linear_regression_"+str(i))

    # %%
    # Hyper parameter tuning
        best_hps, best_model, tuner, history = hyper_model(X_train,y_train, X_val, y_val,
                                                        EPOCH, factor,
                                                        folder_name = str('keras_hyperparameter_aug_'+str(augmentation)+"_res_"+
                                                                          str(residual_subtract)+"_mfcsum_"+str(tot_mfc_scaler)+
                                                                          "_blind_"+str(blind_flip)))
        model_evaluate = model_eval(best_model, X_test, y_test, scaler_coords, X_train, y_train, X_val, y_val, history)
        model_metric = model_comparison(model_metric, model_evaluate, augmentation = str(augmentation), 
                                        residual_subtract = str(residual_subtract), mfc_sum_scaler = str(tot_mfc_scaler), 
                                        blind_flip = str(blind_flip), label = "nn_hyper_model_"+str(i))
        model_metric.to_csv("model_performance.csv", index=False)
        break
# %%