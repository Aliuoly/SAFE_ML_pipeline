The TabML pipeline effectively gauges the effectiveness of both linear and nonlinear models and automates many common tasks in tabular data ML such as outlier removal, data scaling, feature selection, feature engineering (using SAFE) and hyperparameter optimization. 
It also avoids data leakage throughout the processes and gauges unbiased model generalizability thanks to the use of nested CV. 

TabML was made with intention for use with small datasets (less than 10000 samples), though if memory and time allows, larger datasets should work just fine.

TabML is a streamlined tabular data machine learning 'pipeline' that does the following:
1. Removes outliers using isolation forest according to specified contamination value
2. Eliminates redundant features with Spearman's coefficient > 0.8 if enabled
3. Log-transforms the target to ease data skewing if enabled
4. Scales features and target according to specified sklearn scaler class
5. Trains a blackbox model (xgboost by default) as wrapper for iterative feature selection by removing unimportant features
6. Further performs feature selection based on 2 possible criteria: a) keep top 'm' features or b) keep top 'percentile' features
7. Trains a surrogate blackbox model with the remaining feature set
8. Fits a SAFE transformer (https://github.com/ModelOriented/SAFE/tree/master) for categorical feature extraction
9. Trains a SAFE model using an interpretable model (default ElasticNet) and the SAFE features
10. Trains another interpretable model also using iterative feature selection - called the 'base' model

In the end, 4 models are trained and evaluated: wrapper, surrogate, SAFE, and base.
All evaluations are done using nested cross-validation, with the inner folds performing hyperparameter optimization using the hyperopt library.


To recreate the environment in conda, run the following commands
conda create -n <name> python=3.8
pip install xgboost==1.7.6 tqdm==4.65.0 scikit-learn==1.3.0 pandas==2.0.3 numpy==1.24.4 matplotlib==3.7.2 hyperopt==0.2.7 shap==0.42.1 safe-transformer==0.0.5


