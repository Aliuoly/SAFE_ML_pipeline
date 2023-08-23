# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 19:22:19 2023

@author: alden
"""


from xgboost import XGBRegressor as xgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, KBinsDiscretizer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import IsolationForest
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.model_selection import cross_val_score, StratifiedKFold, GroupKFold, KFold, BaseCrossValidator
import shap
from tqdm import tqdm
from SafeTransformer import SafeTransformer
from copy import deepcopy
MAX_ITER = 100000  # maximum number of iteration for sklearn models that take this as an argument (e.g., ElasticNet)




class StratifiedKFoldForContinuous(BaseCrossValidator):
    
    '''
    Generates cross validation folds as such:
        1. discretize the target into bins
        2. generate folds such that each bin is roughly equally
            represented in each fold
    If the initial number of bins lead to any single bin containing less than n_splits samples,
    then reduce the number of bins until satisfied.
    
    '''
    
    
    
    def __init__(self, n_splits=5, encode = 'ordinal', strategy = 'quantile', shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.encode = encode
        self.strategy = strategy

    def split(self, X, y, groups=None):
        # Convert the continuous target variable into bins
        n_bins = int(1 + np.log2(len(y))) #Freedman–Diaconis rule
        kbins = KBinsDiscretizer(n_bins=n_bins, encode=self.encode, strategy=self.strategy, subsample = 200000)
        y = y.values.reshape(-1,1)
        y_binned = kbins.fit_transform(y).ravel()
        y_binned_int = y_binned.astype(int)
        class_counts = np.bincount(y_binned_int)

        # check if the minimum count is less than n_splits. if so, make less bins
        while np.min(class_counts) < self.n_splits:
            n_bins = n_bins - 1
            kbins = KBinsDiscretizer(n_bins=n_bins, encode=self.encode, strategy=self.strategy, subsample = 200000)
            y_binned = kbins.fit_transform(y).ravel()
            y_binned_int = y_binned.astype(int)
            class_counts = np.bincount(y_binned_int)
            if n_bins == 1: #this would just be normal K fold
                break
            
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        # save bins and bin widths as attributes
        self.bins = kbins.bin_edges_[0]  # bin edges
        self.bin_widths = np.diff(self.bins)  # bin widths
        return cv.split(X, y_binned)

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits

class TabML:
    def __init__(
            self, 
            data, groups = None, scaler = StandardScaler(),
            blackbox_model = None, blackbox_space = None, blackbox_space_integer_parameters = [None], #concerning blackbox models
            interpretable_model = None, interpretable_space = None, interpretable_space_integer_parameters = [None], #convering interpretable models
            contamination=0, filter_feature_selection=False, iterative_feature_selection = False, m = np.inf, percentile = 0, target_transform=None, #concerning framework parameters
            outer_cv = StratifiedKFoldForContinuous(n_splits = 10), inner_cv = StratifiedKFoldForContinuous(n_splits = 5), max_evals=200,  #concerning model evaluation
            ): 
        '''
        Initializes the SAFE ML pipeline

        Parameters
        ----------
        data : a pandas dataframe with populated column names of size (n_samples, n_features+1)
            contains both the features and target, where the target is held in the last columns
        groups : ndarray of size (n_samples,). Default = None
            only used if the cross validator is group based - such as GroupKFold
            contains group labels for the provided dataset 'data'
        scaler : sklearn scaler. Default = StandardScaler()
            used to scale/normalize the data before fitting
        blackbox_model : a regression model class, such as sklearn's ElasticNet or xgboost's XGBRegressor
            the blackbox model used as 
            1. the wrapper for feature selection, and
            2. the surrogate model after feature selection for the SAFE ML framework
        blackbox_space : type 'set' of Apply object of hyperopt.pyll.base module
            e.g., for a single hyperparameter,
            space = {hp.quniform('n_estimators', 50, 200, 1)}
            The hyperparameter space to search through during hyperparameter optimization
        blackbox_space_integer_parameters : list of strings
            contains the hyperparameters in the optimization space that needs to have type 'int'
            e.g., the 'n_estimators' hyperparameter for xgboost regressors
        interpretable_model : a regression model class, such as sklearn's ElasticNet or xgboost's XGBRegressor
            the interpretable model used as 
            1. the baseline to compare to SAFE results
            2. the model to be trained using SAFE extracted features
        interpretable_space : type 'set' of Apply object of hyperopt.pyll.base module
            e.g., for a single hyperparameter,
            space = {hp.quniform('n_estimators', 50, 200, 1)}
            The hyperparameter space to search through during hyperparameter optimization
        interpretable_space_integer_parameters : list of strings
            contains the hyperparameters in the optimization space that needs to have type 'int'
            e.g., the 'n_estimators' hyperparameter for xgboost regressors   
        contamination : float or 0. Default = 0
            contamination level used in Isolation Forest for outlier removal. 
            If 0, Isolation Forest is not applied.
        filter_feature_selection : bool. Default = False
            if True, applies Spearman's pair wise correlation for initial feature selection.
        outer_cv : sklearn cross validator or any cross validation fold maker that behaves like sklearn cross validator
            Default: StratifiedKFoldCrossValidator with n_splits = 10
            cross validator used in the outer folds
        inner_cv : sklearn cross validator or any cross validation fold maker that behaves like sklearn cross validator
            Default: StratifiedKFoldCrossValidator with n_splits = 5
            cross validator used in the inner folds (for hyperparameter optimization)
        max_evals : int. Default = 200
            maximum number of evaluations for hyperparameter optimization in the inner folds.
        m : int or np.inf. Default = np.inf 
            number of features to keep from the wrapper model.
            default will retain all features
        percentile : float in range [0,1). Default = 0
            percentile used to determine which features are retained - if 0, all features are retained, if 1 (error), no features are retained.
            default will retain all features
        target_transform : None or 'log'. Default = None
            type of target transformation to apply.

        Returns
        -------
        None.

        '''
        self.data = data
        self.groups = groups
        self.scaler = scaler
        self.target_name = data.columns[-1]
        self.feature_names = data.columns[:-1].tolist()
        self.fitted = False
        self.blackbox_model = blackbox_model
        self.blackbox_space = blackbox_space
        self.blackbox_space_integer_parameters = blackbox_space_integer_parameters
        self.interpretable_model = interpretable_model
        self.interpretable_space = interpretable_space
        self.interpretable_space_integer_parameters = interpretable_space_integer_parameters
        
        # default states if not provided
        if blackbox_model is None:
            print("\nEither blackbox model class or hyperparameter space or both are not provided, will use default XGBoost instead")
            self.blackbox_model = xgb
            self.blackbox_space = {
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
            self.blackbox_space_integer_parameters = ['n_estimators','max_depth','min_child_weight']
        else:
            self.blackbox_model = blackbox_model
            self.blackbox_space = blackbox_space
            self.blackbox_space_integer_parameters = blackbox_space_integer_parameters
        
        if interpretable_model is None or interpretable_space is None:
            print("\nEither interpretable model class or hyperparameter space or both are not provided, will use default elastic net instead")
            self.interpretable_model = ElasticNet
            self.interpretable_space = {
                'alpha': hp.uniform('alpha', 0.0, 1.0),
                'l1_ratio': hp.uniform('l1_ratio', 0.0, 1.0),
            }
            self.interpretable_space_integer_parameters = [None]
        else:
            self.interpretable_model = interpretable_model
            self.interpretable_space = interpretable_space
            self.interpretable_space_integer_parameters = interpretable_space_integer_parameters
        
        self._preprocessed = False
        self.wrapper_model = None
        self.surrogate_model = None
        self.base_model = None
        self.SAFE_model = None
        self.important_feature_names = None
        self.selected_feature_names = None
        self.base_feature_names = None
        self.wrapper_feature_names = None
        self.SAFE_transformer = None
        
        self.contamination = contamination
        if percentile < 0 or percentile >= 1:
            raise ValueError(f"percentile must be a decimal in range [0,1), got {percentile} instead")
        elif (percentile == 0 and m != np.inf) or (percentile != 0 and m != np.inf):
            raise ValueError("either percentile or m should be specified, not both")
        self.m = m
        self.percentile = percentile
        
        self.outer_cv = outer_cv
        self.inner_cv = inner_cv
        
        if isinstance(outer_cv, GroupKFold):
            if groups is None:
                raise ValueError("GroupKFold cross validator is used but no group information is provided")
            self.groups = groups
        else:
            self.groups = None
            
        self.max_evals = max_evals
        self.target_transform = target_transform
        
        self.filter_feature_selection = filter_feature_selection
        self.iterative_feature_selection = iterative_feature_selection
        
        self.y_scaler = None
        
    def fit_model(self, model_key, verbose = True, penalty = 2):
        
        '''
        Fits the model as designated by model_key
        
        Parameters
        ----------
        model_key : str = 'wrapper','surrogate','base', or 'SAFE'
            designates which model to train along the pipeline
            wrapper is used for iterative and subsequent feature selection
            surrogate is used to fit the SAFE transformer for generating the SAFE dataset
            base is the basic interpretable model trained on final dataset
            SAFE is the interpretable model trained on the SAFE dataset
        verbose : bool
            whether or not to display progress as fitting takes place
        penalty : float
            only applicable when model_key is 'SAFE'
            the penalty value used when fitting the SAFE transformer
            
        Returns
        -------
        return_model : estimator
            the fitted model of the designated key
        return_data : pandas DataFrame
            the data used to train the return_model
        return_scaler : sklearn scaler
            fitted sklearn scaler used to scale the original data to the returned data
            for use in inverse transforming the returned data if desired
        
        
        '''
        
        
        self.verbose = verbose
        if self._preprocessed is False:
            print("preprocessing data")
            self._preprocess(self.data)
            
        print("getting relevant dataset")
        
        if model_key == "surrogate":
            if self.wrapper_model is None:
                raise AttributeError("wrapper model has not yet been fitted - run 'fit_model' with 'wrapper' as model key first")
            elif self.feature_importances is None or self.important_feature_names is None:
                raise AttributeError("feature importances has not yet been evaluated - run 'evaluate_model' with 'wrapper' as model key first")
                
        elif model_key == "SAFE":
            if self.surrogate_model is None:
                raise AttributeError("surrogate model has not yet been fitted - run 'fit_model' with 'surrogate' as model key first")
            elif self.SAFE_transformer is None or penalty == self.penalty: #if notyet fitted or using same penalty as last time
                self.penalty = penalty
                print("fitting SAFE transformer using surrogate model")
                return_data, return_scaler = self.get_relevant_data(model_key = "surrogate")
                X, y = return_data.iloc[:,:-1],return_data.iloc[:,-1]
                self.SAFE_transformer = self._get_transformer(self.surrogate_model, X = X, penalty = penalty)
                
            
        return_data, return_scaler = self.get_relevant_data(model_key = model_key)
        X, y = return_data.iloc[:,:-1],return_data.iloc[:,-1]
        if self.groups is None:
            groups = None
        else:
            groups = self.groups[self.non_outlier_indeces]
        if verbose: print(f"\nTraining {model_key} model")
        if model_key == 'wrapper':
            if self.iterative_feature_selection == True:
                self.wrapper_model, self.wrapper_feature_names = self._iterative_feature_selection(self.blackbox_model, self.blackbox_space, X, y, groups)
                return_data = pd.concat((X[self.wrapper_feature_names], y), axis = 1)
            else:
                self.wrapper_model = self._hyperopt_model(self.blackbox_model, self.blackbox_space, X, y, groups)
                self.wrapper_feature_names = X.columns.tolist()
                return_model = self.wrapper_model
            
            return_model = self.wrapper_model
            
        elif model_key == "surrogate":
            if self.m == np.inf and self.percentile == 0:
                #if no feature selection is to be carried out based on number of features to keep or percentile of important features
                #then the surrogate is the blackbox model
                self.surrogate_model = self.wrapper_model
                return_model = self.surrogate_model
            else:
                self.surrogate_model = self._hyperopt_model(self.blackbox_model, self.blackbox_space, X, y, groups)
                return_model = self.surrogate_model
                    
        elif model_key == "base":
            if self.iterative_feature_selection == True:
                self.base_model, self.base_feature_names = self._iterative_feature_selection(self.interpretable_model, self.interpretable_space, X, y, groups)
                return_model = self.base_model
                return_data = pd.concat((X[self.base_feature_names], y), axis = 1)
            else:
                self.base_model = self._hyperopt_model(self.interpretable_model, self.interpretable_space, X, y, groups)
                self.base_feature_names = X.columns.tolist()
                return_model = self.base_model
            
            
        elif model_key == "SAFE":
            self.SAFE_model = self._hyperopt_model(self.interpretable_model, self.interpretable_space, X, y, groups)
            return_model = self.SAFE_model
            
        # get return data and scaler again in case features have been selected, etc. 
        return_data, return_scaler = self.get_relevant_data(model_key = model_key)
        return return_model, return_data, return_scaler
    
    def _iterative_feature_selection(self,model,space,X,y,groups):
        kept_feature_names = X.columns.tolist()
        while True:
            trained_model = self._hyperopt_model(model, space, X, y, groups)
            feature_importances = self._get_shap_values(trained_model,X,X).mean(axis=0) #no worries for leakage here - only goal is to identify zero importance features, not get true value of feature importance
            if any(feature_importances == 0):
                kept_feature_names = [name for i,name in enumerate(X.columns) if np.abs(feature_importances[i]) != 0]
                X = X[kept_feature_names]
                if self.verbose:print(f"\n{len(kept_feature_names)} features selected in this round")
            else:
                if self.verbose: print(f"\n{len(kept_feature_names)} importance features selected. Endding iterative selection")
                break
            
        return trained_model, kept_feature_names
    
    def evaluate_model(self, model_key, model_data, verbose = True):
        '''
        Evaluates the fitted model as designated by model_key
        
        Parameters
        ----------
        model_key : str = 'wrapper','surrogate','base', or 'SAFE'
            designates which model to evaluate
        model_data : pandas DataFrame
            dataset used in fitting the designated model (to do: when model_data is not provided, get the data using self.get_relevant_data)
        verbose : bool
            whether or not to display progress as evaluation takes place

            
        Returns
        -------
        mse: list
            mean squared errors from the outer cross validation folds
        r2 :list
            r2 score from the outer cross validation folds
        
        
        '''
        
        
        
        
        X, y = model_data.iloc[:,:-1], model_data.iloc[:,-1]
        if verbose: print(f"evaluating {model_key} model")
        
        if model_key == 'wrapper':
            mse, r2, self.important_feature_names = self._nested_cv(self.blackbox_model, self.blackbox_space, X[self.wrapper_feature_names], y, get_shap = True)
        elif model_key == 'surrogate':
            mse, r2 = self._nested_cv(self.blackbox_model, self.blackbox_space, X, y, get_shap = False)
        elif model_key == "base" or "SAFE":
            mse, r2 = self._nested_cv(self.interpretable_model, self.interpretable_space, X, y, get_shap = False)
        else:
            raise ValueError(f"valid model keys are 'blackbox' and 'interpretable' only, but got {model_key} instead")
        
        return mse, r2
        
        
        
    def fit(self, verbose = True, penalty = 2):
        '''
        fits the SAFE ML pipeline and generates each sub models and evaluates them
        1. wrapper - blackbox model trained on all the filtered features and with which iterative feature selection is based on
        2. surrogate - blackbox model trained on the top "m" or top 'percentile' features from wrapper
        3. base - interpretable model trained on the top "m" or top 'percentile' features from wrapper
        4. SAFE - interpretable model trained on the SAFE features extracted from surrogate

        Parameters
        ----------
        verbose : bool
            whether or not to display progress as fitting takes place
        penalty : float
            only applicable when model_key is 'SAFE'
            the penalty value used when fitting the SAFE transformer
        

        Returns
        -------
        model_summary: dictionary
            dictionary containing the nested cross validation scores of each model of each fold

        '''
        # Initialize all datasets to be saved
        self.verbose = verbose
        
        # Save target and feature names for later use
        self.target_name = self.data.columns[-1]
        self.feature_names = self.data.columns[:-1].tolist()
        
        _, data, _ = self.fit_model(model_key = "wrapper", verbose = verbose)
        wrapper_mse, wrapper_r2 = self.evaluate_model(model_key = "wrapper", model_data = data, verbose = verbose)
        
        _, data, _ = self.fit_model(model_key = "surrogate", verbose = verbose)
        if self.m == np.inf and self.percentile == 0:
            #if no feature selection is to be carried out based on number of features to keep or percentile of important features
            #then the surrogate is the blackbox model
            surrogate_mse, surrogate_r2 = wrapper_mse, wrapper_r2
        else:
            surrogate_mse, surrogate_r2 = self.evaluate_model(model_key = "surrogate", model_data = data, verbose = verbose)
        
        _, data, _ = self.fit_model(model_key = "base", verbose = verbose)
        base_mse, base_r2 = self.evaluate_model(model_key = "base", model_data = data, verbose = verbose)
        
        _, data, _ = self.fit_model(model_key = "SAFE", penalty = penalty, verbose = verbose)
        SAFE_mse, SAFE_r2 = self.evaluate_model(model_key = "SAFE", model_data = data, verbose = verbose)

        
        
        # save result summary
        self.model_summary = {
            'wrapper': {
                'mse': wrapper_mse,
                'r2': wrapper_r2,
            },
            'surrogate': {
                'mse': surrogate_mse,
                'r2': surrogate_r2
            },
            'base': {
                'mse': base_mse,
                'r2': base_r2
            },
            'SAFE': {
                'mse': SAFE_mse,
                'r2': SAFE_r2
            }
        }
        
        return self.model_summary
        
    def get_relevant_data(self, model_key):
        '''
        
        generates the relevant dataset to the model as designated by model_key
        
        
        Parameters
        ----------
        model_key : str = 'wrapper', 'surrogate', 'base', or 'SAFE'
            which specific use dataset is return.
        raw_data : pandas DataFrame
            the same DataFrame as passed to the fit or fit_n_times method.

        Raises
        ------
        ValueError
            if no valid key was provided, raise error.

        Returns
        ------
        return_data : pandas DataFrame
            the dataset (X and y) used to train the model as designated by model_key
        return_scaler : sklearn scaler
            fitted scaler used to scale the dataset to the scale used during training

        '''
        if self._preprocessed:
            filtered_column_names = self.filtered_feature_names.copy()
            filtered_column_names.append(self.target_name)
            filtered_data = self.data.loc[self.non_outlier_indeces,filtered_column_names]
            if self.target_transform == 'log':
                 filtered_data[self.target_name] = np.log(filtered_data[self.target_name])

            return_data = pd.DataFrame()
            filtered_data = pd.DataFrame(data = self.scaler.transform(filtered_data), columns = filtered_column_names)
            return_scaler = self._get_return_scaler(model_key)
            
            if model_key == 'wrapper':
                column_names = self.wrapper_feature_names.copy()
                column_names.append(self.target_name)
            elif model_key == 'base':
                column_names = self.base_feature_names.copy()
                column_names.append(self.target_name)
            elif model_key == 'surrogate':
                if self.important_feature_names is not None:
                    if self.selected_feature_names is None:
                        self._get_selected_feature_names()
                    column_names = self.selected_feature_names.copy()
                    column_names.append(self.target_name)
                else:
                    raise AttributeError("feature selection has not taken place - use the method 'evaluate_model' with model_key 'wrapper' first")
            elif model_key == 'SAFE':
                if self.SAFE_transformer is not None:
                    return_data = self.SAFE_transformer.transform(filtered_data[self.selected_feature_names])
                    return_data[self.target_name] = filtered_data[self.target_name]
                    #return_stats = stats
                    return return_data, return_scaler
                else:
                    raise AttributeError("SAFE model was not fitted")
            else:
                raise ValueError("Model key not recognized. \nValid keys are 'wrapper', 'surrogate', 'base', or 'SAFE'")
        else:
            raise AttributeError("No fitting has taken place; cannot get relevant data")
            
        return_data = filtered_data[column_names].copy()

        return return_data, return_scaler
    
    def _get_return_scaler(self, model_key):
        '''Generates the scaler used to scale the dataset used during training of the model designated by model_key'''
        
        

            
        if model_key == 'SAFE':
            return_column_names = []
        else:
            if model_key == 'wrapper':
                if isinstance(self.wrapper_feature_names, pd.Index):
                    self.wrapper_feature_names = self.wrapper_feature_names.tolist()
                return_column_names = self.wrapper_feature_names.copy()
            elif model_key == 'base':
                if isinstance(self.base_feature_names, pd.Index):
                    self.base_feature_names = self.base_feature_names.tolist()
                return_column_names = self.base_feature_names.copy()
            elif model_key == 'surrogate':
                if self.selected_feature_names is None:
                    self._get_selected_feature_names()
                return_column_names = self.selected_feature_names.copy()
            else:
                raise ValueError("model_key not supported. Only 'wrapper','surrogate','base', or 'SAFE' are valid model keys")
        

                
        return_column_names.append(self.target_name)  
        data = self.data.loc[self.non_outlier_indeces,return_column_names].copy()
        return_scaler = deepcopy(self.scaler)
        return_scaler.fit_transform(data)
            
        return return_scaler
            
        
        
    def _get_selected_feature_names(self):
        ''' Generates and saves the selected feature names based on 'm' or 'percentile' criteria set during initialization'''
        if self.m == np.inf and self.percentile == 0:
            #if no feature selection is to be carried out based on number of features to keep or percentile of important features
            #then the surrogate is the blackbox model
            self.selected_feature_names = self.wrapper_feature_names
        elif self.percentile == 0:
            if len(self.important_feature_names) > self.m:
                self.selected_feature_names = self.important_feature_names[:self.m]
            else:
                self.selected_feature_names = self.important_feature_names
        else:
            threshold = np.percentile(self.feature_importances,self.percentile*100)
            crit = self.feature_importances.values>threshold
            self.selected_feature_names = [name for i,name in enumerate(self.important_feature_names) if crit[i]]
            
        print(f"\n{len(self.selected_feature_names)} most important features selected")

    
    def _preprocess(self, data):
        '''
        preprocesses the given data by (if configured to do so)
        1. removing collinear features based on Spearman coefficients (>0.80)
        2. removing outliers using isolation forest
        3. log transforms target
        4. scales features and target using scaler provided during initialization
        '''
        
        #get dataset into workable forms
        X, y = data.iloc[:,:-1], data.iloc[:,-1]
        column_names = data.columns.tolist()
        target_name = column_names[-1]
        feature_names = column_names[:-1]
        #0. remove outlier if told to
        clean_idx = range(len(y))
        outlier_idx = []
        if self.contamination != 0:
            outlier_remover = IsolationForest(contamination=self.contamination)
            #fit to training data to avoid testing data leakage
            outliers = outlier_remover.fit_predict(X)               
            clean_idx = outliers != -1
            outlier_idx = outliers == -1
            X, y = X.iloc[clean_idx, :], y.iloc[clean_idx]  
            if self.verbose: print(f"{len(y)} data points kept")
            
        #1. filter based feature selection
        if self.filter_feature_selection:
            corr_matrix = X.corr(method = 'spearman').abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
            X = X.drop(X[to_drop], axis=1)
            feature_names = X.columns.tolist()
            column_names = X.columns.tolist().append(target_name)
            if self.verbose: print(f"{len(feature_names)} features kept after filter based feature selection")
        
        #2. transform target first if indicated
        if self.target_transform == 'log':
            y = np.log(y)
            
            
        #3. scale features to zero mean and unit variance 
        #recombine X and y for processing
        data = pd.concat((X,y),axis = 1)
        self.scaler.fit_transform(data)
        
        # save useful information in class
        self.non_outlier_indeces = clean_idx
        self.outlier_indeces = outlier_idx
        self.filtered_feature_names = feature_names
        self.wrapper_feature_names = feature_names
        self.base_feature_names = feature_names
        
        self._preprocessed = True
    
    
    def _inverse_transform_y(self, y):
        
        '''
        inverse transform/scale the target/prediction using the saved scaler back to original magnitude

        Parameters
        ----------
        y : pandas series or numpy ndarray
            the target/prediction to be scaled back to the original magnitude

        Returns
        -------
        unscaled_y : pandas series or numpy ndarray
            the target/prediction already scaled back to the original magnitude

        '''
        
        #1. standardscaler inverse transform
        if self.y_scaler is None:
            original_y = self.data.loc[self.non_outlier_indeces, self.target_name].values.reshape(-1,1)
            if self.target_transform == 'log':
                original_y = np.log(original_y)
            self.y_scaler = deepcopy(self.scaler)
            self.y_scaler.fit(original_y)
        
        if isinstance(y, np.ndarray):
            y = y.reshape(-1,1)
        else:
            y = y.values.reshape(-1,1)
            
        unscaled_y = self.y_scaler.inverse_transform(y)
        
        #2. target transform if applicable
        if self.target_transform == 'log':
            unscaled_y = np.exp(y)
            
        return unscaled_y
    
    
    def _get_metrics(self,model,X,y):
        
        '''
        get mse and r2 scores using the provided model, features, and ground truth target

        Parameters
        ----------
        model : estimator model capable of .predict()
            the estimator model to be evaluated
        X : pandas DataFrame
            The 
        y : pandas series or numpy ndarray
            the ground truth target

        Returns
        -------
        y : pandas series or numpy ndarray
            the target/prediction already scaled back to the original magnitude
            
        '''
        
        
        y = self._inverse_transform_y(y)
        prediction = model.predict(X)
        prediction = self._inverse_transform_y(prediction)
        
        mse = mean_squared_error(y,prediction)
        r2 = r2_score(y,prediction)
        
        return mse, r2
    
    def _get_transformer(self, surrogate, penalty, X):
        """Instantiate and fit a SafeTransformer."""
        transformer = SafeTransformer(surrogate, penalty=penalty)
        transformer.fit(X, verbose=False)
        return transformer
    
    
    def _nested_cv(self, model_class, model_space, X, y, get_shap=False):
        """Performs nested cross validation on the designated model class and hyperparameter space."""
        
        feature_importances = pd.DataFrame(columns=X.columns)
        shap_values_list = []
        outer_mse_scores = []
        outer_r2_scores = []
        
        # Outer cross validation
        if self.verbose: pbar = tqdm(total=self.outer_cv.n_splits, desc="Outer CV")
        
        # use group information is applicable and provided
        if self.groups is not None:
            groups = self.groups[self.non_outlier_indeces]
        else:
            groups = self.groups
            
        for train_idx, valid_idx in self.outer_cv.split(X, y, groups = groups):
            
            X_train, X_valid, y_train, y_valid = self._get_train_valid(X, y, train_idx, valid_idx)
            
            if self.groups is not None:
                groups_train = groups[train_idx]
            else:
                groups_train = None
                
            tuned_model = self._get_tuned_model(model_class, model_space, X_train, y_train, groups_train)
        
            if get_shap:
                shap_values = self._get_shap_values(tuned_model, X_train, X_valid)
                feature_importances.loc[len(feature_importances)] = np.mean(np.abs(shap_values), axis=0) #basically append
                shap_values_list.append(shap_values)
        
            mse, r2 = self._get_metrics(tuned_model, X_valid, y_valid)
            outer_mse_scores.append(mse)
            outer_r2_scores.append(r2)
            
            # Update the progress bar and postfix
            if self.verbose: 
                pbar.update(1)
                pbar.set_postfix({'r2_mean': np.mean(outer_r2_scores)}, refresh=True)
            
        # get nested CV results
        #mse_mean, mse_std = np.mean(outer_mse_scores), np.std(outer_mse_scores, ddof=1)
        r2_mean, r2_std = np.mean(outer_r2_scores), np.std(outer_r2_scores, ddof=1)
        
        
        if self.verbose: 
            pbar.set_postfix({'r2': f"{r2_mean:.3f} ({r2_std:.3f})"}, refresh=True)
            pbar.close()

        if get_shap:
            ranked_features = self._get_ranked_features(feature_importances)  
            self.shap_values = shap_values_list
            return outer_mse_scores, outer_r2_scores, ranked_features
            
        return outer_mse_scores, outer_r2_scores
    
    
    def _get_train_valid(self, X, y, train_idx, valid_idx):
        """Create training and validation datasets."""
        return X.iloc[train_idx, :], X.iloc[valid_idx, :], y.iloc[train_idx], y.iloc[valid_idx]
    
    
    def _get_tuned_model(self, model_class, model_space, X_train, y_train, groups = None):
        """Tune the model hyperparameters and fit the model."""
        return self._hyperopt_model(model_class, model_space, X_train, y_train, groups)
    
    
    def _get_shap_values(self, tuned_model, X_train, X_valid):
        """Compute SHAP values."""
        try:
            explainer = shap.Explainer(tuned_model)
            return explainer.shap_values(X_valid)
        except Exception:
            try:
                explainer = shap.KernelExplainer(tuned_model, X_train)
                return explainer.shap_values(X_valid)
            except Exception:
                try:
                    explainer = shap.LinearExplainer(tuned_model, X_train)
                    return explainer.shap_values(X_valid)
                except Exception as e:
                    raise RuntimeError(f"Neither shap.Explainer, shap.KernelExplainer, or shap.LinearExplainer were applicable due to {e}")
        
    
    
    def _get_ranked_features(self, feature_importances):
        """Rank features by their importance."""
        mean_feature_importance = feature_importances.mean()
        self.feature_importances = mean_feature_importance
        ranked_features = mean_feature_importance.sort_values(ascending=False)
        return ranked_features.index.tolist()


    def _define_hyperopt_objective(self, model_class, X, y, groups = None):
        """Define the objective function for hyperparameters optimization."""
        def hyperopt_objective(params):
            
            # fix hyperparameter type if necessary - e.g. xgboost has this issue
            if model_class == self.blackbox_model:
                integer_parameters = self.blackbox_space_integer_parameters
            else:
                integer_parameters = self.interpretable_space_integer_parameters
            
            
            for param in integer_parameters:
                if param in params:
                    params[param] = int(params[param])
                    
           # Try to instantiate model with 'max_iter' parameter. If it's not accepted, try without it.     
            try:
                model = model_class(**params, max_iter=MAX_ITER)
            except TypeError:
                model = model_class(**params)
    
    
            # Cross validate model and compute score
            cv = self.inner_cv
            score = -cross_val_score(model, X, y, cv=cv, groups = groups, n_jobs=-1)
            
            # Return a dictionary with status, mean loss, and loss variance
            return {
                'status': STATUS_OK,
                'loss': score.mean(),
                'loss_variance': np.var(score, ddof=1)
            }
        return hyperopt_objective
    
    
    def _perform_hyperopt(self, objective, hyperparameter_space, verbose):
        """Perform hyperparameters optimization."""
        trials = Trials()
        best = fmin(
            fn=objective,
            space=hyperparameter_space,
            algo=tpe.suggest,
            max_evals=self.max_evals,
            trials=trials,
            verbose=verbose
        )
        return best
    
    
    def _construct_and_fit_model(self, model_class, best_params, X, y):
        """Construct and fit the model with the best found parameters."""
        
        # fix hyperparameter type if necessary - e.g. xgboost has this issue
        if model_class == self.blackbox_model:
            integer_parameters = self.blackbox_space_integer_parameters
        else:
            integer_parameters = self.interpretable_space_integer_parameters
        for param in integer_parameters:
            if param in best_params:
                best_params[param] = int(best_params[param])

        # Try to instantiate model with 'max_iter' parameter. If it's not accepted, try without it.    
        try:
            if 'XGB' in model_class.__name__:
                model = model_class(**best_params)
            else:
                model = model_class(**best_params, max_iter=MAX_ITER)
        except TypeError:
            model = model_class(**best_params)
                
        model.fit(X, y)
        return model
    
    
    def _hyperopt_model(self, model_class, hyperparameter_space, X, y, groups = None):
        """Define objective, perform hyperparameter optimization, construct and fit the model."""
        hyperopt_objective = self._define_hyperopt_objective(model_class, X, y, groups)
        best_params = self._perform_hyperopt(hyperopt_objective, hyperparameter_space, verbose = self.verbose)    
        model = self._construct_and_fit_model(model_class, best_params, X, y)
        return model

                

        
            
        