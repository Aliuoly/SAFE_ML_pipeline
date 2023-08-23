# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 22:32:34 2023

@author: alden
"""
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import PredictionErrorDisplay
from sklearn.model_selection import cross_val_predict, KFold
from TabML import TabML
from hyperopt import hp
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

# get some toy dataset
X, y, true_coefs = make_regression(n_samples = 100, n_features = 50, noise = 0.1, coef = True)
data = pd.DataFrame(data = X, columns = [f'x{i+1}' for i in range(X.shape[1])])
data['y'] = y

# define which model to use as blackbox (wrapper and surrogate)
# also define the hyperparameter optimization space using the hyperopt library
blackbox_model = XGBRegressor
blackbox_space = {
    'n_estimators': hp.quniform('n_estimators', 50, 500, 1),
    'learning_rate': hp.loguniform('learning_rate', 0.01, 0.3),
    'max_depth': hp.quniform('max_depth', 1,10,1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    'gamma': hp.uniform('gamma', 0.1, 1),
    'reg_alpha': hp.uniform('reg_alpha', 0, 1),
    'reg_lambda': hp.uniform('reg_lambda', 0, 3),
    'subsample': hp.uniform('subsample', 0.5, 1),
    'min_child_weight': hp.quniform('min_child_weight', 1, 5, 1),
    'verbosity' : 0,
    #'tree_method': 'gpu_hist',
    #'gpu_id': 0
}

# define which hyperparameters are supposed to be integers
blackbox_space_integer_parameters = ['n_estimators','max_depth','min_child_weight']


# define which model to use as interpretable model (base and/or SAFE)
# also define the hyperparameter optimization space using the hyperopt library
interpretable_model = ElasticNet
interpretable_space = {
    'alpha': hp.uniform('alpha', 0.0, 1.0),
    'l1_ratio': hp.uniform('l1_ratio', 0.0, 1.0),
}

# define which hyperparameters are supposed to be integers
interpretable_space_integer_parameters = [None]

# define the outer and inner cross validators
outer_cv = KFold(n_splits = 10)
inner_cv = KFold(n_splits = 5)

# initiate the TabML_model
TabML_model = TabML(
    data = data, 
    groups = None, # no group labels. Not applicable anyways if using non Group based CV. 
    blackbox_model = blackbox_model,
    blackbox_space = blackbox_space,
    blackbox_space_integer_parameters = blackbox_space_integer_parameters,
    interpretable_model = interpretable_model,
    interpretable_space = interpretable_space,
    interpretable_space_integer_parameters = interpretable_space_integer_parameters,
    scaler = StandardScaler(), #use StandardScaler as data scaler
    contamination = 0, # do not perform outlier removal using Isolation Forest
    outer_cv = outer_cv, # cross validator used in outer folds
    inner_cv = inner_cv, # cross validator used in inner folds
    m = np.inf, # do not select the most important 'm' features - keep all features according to this criteria
    percentile = 0, # do not select the most important 'percentile'*100% of features
    max_evals = 50, # allow 50 evaluations for hyperparameter optimization in the inner CV folds
    filter_feature_selection = True, # perform filter based feature selection based on Spearman coefficient between features
    iterative_feature_selection = True, # perform iterative feature selection by eliminating all features with 0 shap values during wrapper model fitting
    target_transform = None # perform no transformation for the target before data scaling
    ) 

# carry out the fitting procedure (or use the .fit method of the TabML class
# first fit the wrapper model and evaluate it
wrapper_model, wrapper_data, wrapper_scaler = TabML_model.fit_model(model_key = 'wrapper') 
wrapper_mse, wrapper_r2 = TabML_model.evaluate_model(model_key = 'wrapper', model_data = wrapper_data)

# fit and evaluate the surrogate model - requires blackbox model be fitted and evaluated
surrogate_model, surrogate_data, surrogate_scaler = TabML_model.fit_model(model_key = 'surrogate')
surrogate_mse, surrogate_r2 = TabML_model.evaluate_model(model_key = 'surrogate', model_data = surrogate_data)

# fit and evaluate the SAFE model - requires surrogate model be fitted but not evaluated
SAFE_model, SAFE_data, SAFE_scaler = TabML_model.fit_model(model_key = 'SAFE')
SAFE_mse, SAFE_r2 = TabML_model.evaluate_model(model_key = 'SAFE', model_data = SAFE_data)

# fit and evaluate the base interpretable model as comparison with the other models
# requires surrogate model be fitted but not evaluated
base_model, base_data, base_scaler = TabML_model.fit_model(model_key = 'base')
base_mse, base_r2 = TabML_model.evaluate_model(model_key = 'base', model_data = base_data)


# generate a summary table of model performances
summary = pd.DataFrame(
    {
     'wrapper':{
        'mse': f'{np.mean(wrapper_mse):.0f} ({np.std(wrapper_mse,ddof=1):.0f})',
        'r2': f'{np.mean(wrapper_r2):.3f} ({np.std(wrapper_r2,ddof=1):.2f})'
        },
    'surrogate':{
        'mse': f'{np.mean(surrogate_mse):.0f} ({np.std(surrogate_mse,ddof=1):.0f})',
        'r2': f'{np.mean(surrogate_r2):.3f} ({np.std(surrogate_r2,ddof=1):.2f})'
        },
    'SAFE':{
        'mse': f'{np.mean(SAFE_mse):.0f} ({np.std(SAFE_mse,ddof=1):.0f})',
        'r2': f'{np.mean(SAFE_r2):.3f} ({np.std(SAFE_r2,ddof=1):.2f})'
        },
    'base':{
        'mse': f'{np.mean(base_mse):.0f} ({np.std(base_mse,ddof=1):.0f})',
        'r2': f'{np.mean(base_r2):.3f} ({np.std(base_r2,ddof=1):.2f})'
        }
    })

print(summary)


# interested in the best model
mses = np.mean([wrapper_mse, surrogate_mse, SAFE_mse, base_mse], axis = 1).tolist()
best_index = mses.index(min(mses))
best_model = [wrapper_model, surrogate_model, SAFE_model, base_model][best_index]
best_data = [wrapper_data, surrogate_data, SAFE_data, base_data][best_index]
best_scaler = [wrapper_scaler, surrogate_scaler, SAFE_scaler, base_scaler][best_index]
best_model_key = ['wrapper','surrogate','SAFE','base'][best_index]
# interested in the original scale dataset used for the best model - use the returned data and scaler
if best_scaler != SAFE_scaler:
    original_scale_best_data = pd.DataFrame(best_scaler.inverse_transform(best_data), columns = best_scaler.feature_names_in_)
else:
    print("original scale best data is not meaningful - SAFE uses extracted categorical features")


# interested in getting a rough visual of prediction vs. true values for the best model
wrapper_model = TabML_model.wrapper_model
y_pred = cross_val_predict(best_model,best_data.iloc[:,:-1],best_data.iloc[:,-1],cv = TabML_model.outer_cv)
y_pred = TabML_model._inverse_transform_y(y_pred)
y = TabML_model._inverse_transform_y(best_data.iloc[:,-1])
resi = y-y_pred
resi = resi
fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
PredictionErrorDisplay.from_predictions(y,y_pred=y_pred,kind="actual_vs_predicted",ax=axs[0])
axs[0].set_title("Actual vs. Predicted values")
PredictionErrorDisplay.from_predictions(y,y_pred=y_pred,kind="residual_vs_predicted",ax=axs[1],random_state=0)
axs[1].set_title("Residuals vs. Predicted Values")
fig.suptitle("Plotting cross-validated predictions")
plt.show()


# interested in the SHAP values of the best performing model
X = best_data.iloc[:,:-1]
try:
    explainer = shap.Explainer(best_model)
except Exception:
    try:
        explainer = shap.KernelExplainer(best_model, X)
    except Exception:
        try:
            explainer = shap.LinearExplainer(best_model, X)
        except Exception as e:
            raise RuntimeError(f"Neither shap.Explainer, shap.KernelExplainer, or shap.LinearExplainer were applicable due to {e}")

shap_values = explainer.shap_values(X)
plt.figure()
shap.summary_plot(shap_values, X)

# if best model is a linear model with attribute .coef_, plot that as well
try:
    coefs = best_model.coef_.tolist() #append a random num as 'target' for scaler to work
    plt.figure()
    plt.barh(width = coefs, y = X.columns)
    plt.title("coefficients for each feature used in the best model")
    plt.figure()
    plt.barh(width = true_coefs[true_coefs != 0], y = data.iloc[:,:-1].columns[true_coefs != 0])
    plt.title("TRUE coefficients for each feature")
except Exception:
    print("best model does not have attribute .coef_")
            