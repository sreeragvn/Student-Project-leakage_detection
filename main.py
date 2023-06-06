# %%
from utils import *
single_leakage, two_leakage = load_data()
# single_leakage_data_spread(single_leakage)=
span = 16048
width = 5233

print(single_leakage.isna().values.any())
# variance_plot(single_leakage)
# %%
# data_check(single_leakage)
# Average are not really close together. should we be worried ?

# %%
# split_xy_save(single_leakage)
# %%
X_train, X_test, X_val, y_train, y_test, y_val, scaler_coords, scaler_flows = load_single_leakage_model_data(span, width)
# %%
scikit_linear_regression(X_train, y_train, X_test, y_test, scaler_coords, X_val, y_val)
# %%
linear_model, history = benchmark_linear_model(X_train, y_train, X_val, y_val)
model_eval(linear_model, X_test, y_test, scaler_coords, X_train, y_train, X_val, y_val, history)
# %%
# Hyper parameter tuning
EPOCH = 1000
factor = 2
best_hps,hyper_model, tuner, history = hyper_model(X_train,y_train, X_val, y_val, EPOCH, factor)

model_eval(hyper_model, X_test, y_test, scaler_coords, X_train, y_train, X_val, y_val, history)
# %%w
hyper_model.summary()