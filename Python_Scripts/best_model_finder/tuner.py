# Import Libraries

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression 
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

class Model_Finder:
    
    """
        This class shall be used to find the model with the best accuracy and AUC score.
    """

    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
        
        self.models = {
                        "RandomForest": RandomForestClassifier(),
                        "KNN": KNeighborsClassifier(),
                        "DecisionTree": DecisionTreeClassifier(),
                        "LogisticRegression": LogisticRegression(),
                        "LinearRegression": LinearRegression(),
                        "AdaBoost": AdaBoostClassifier(),
                        "GradientBoosting": GradientBoostingClassifier(),
                        "SVC": SVC(probability=True),
                        "GaussianNB": GaussianNB(),
                        "XGBoost": XGBClassifier()
                    }

    def get_best_params_for_model(self, model_name, param_grid, train_x, train_y):
        
        """
            Generic method to get the best parameters for a given model.
            model_name: The name of the model to be tuned.
            param_grid: The parameter grid for the model.
            train_x: Training features.
            train_y: Training labels.
            Output: The model with the best parameters.
            On Failure: Raise Exception.
        """
        
        self.logger_object.log(self.file_object, 
                            f'Entered the get_best_params_for_{model_name} method of the Model_Finder class')

        try:
            model = self.models[model_name]
            grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, verbose=3)
            grid.fit(train_x, train_y)
            
            best_params = grid.best_params_
            self.logger_object.log(self.file_object, 
                                f'{model_name} best params: {best_params}')

            # Create a new model with the best parameters
            best_model = model.__class__(**best_params)
            best_model.fit(train_x, train_y)
            self.logger_object.log(self.file_object, 
                                f'Exited the get_best_params_for_{model_name} method of the Model_Finder class')

            return best_model
        
        except Exception as e:
            self.logger_object.log(self.file_object, 
                                f'Exception occurred in get_best_params_for_{model_name} method of the Model_Finder class.'
                                f'Exception message: {str(e)}')
            
            self.logger_object.log(self.file_object, 
                                f'{model_name} Parameter tuning failed.' 
                                f'Exited the get_best_params_for_{model_name} method of the Model_Finder class')
            
            raise Exception()

    def get_best_model(self, train_x, train_y, test_x, test_y):
        
        """
            Find out the Model which has the best AUC score.
            Output: The best model name and the model object.
            On Failure: Raise Exception.
        """
        
        self.logger_object.log(self.file_object,
                            'Entered the get_best_model method of the Model_Finder class')

        try:
            models_scores = {}
            
            param_grids = {
                            "RandomForest": {
                                                "n_estimators": [10, 50, 100, 130],
                                                "criterion": ['gini', 'entropy'],
                                                "max_depth": range(2, 4, 1),
                                                "max_features": ['auto', 'log2']
                                            },
                            "KNN": {
                                        'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                                        'leaf_size': [10, 17, 24, 28, 30, 35],
                                        'n_neighbors': [4, 5, 8, 10, 11],
                                        'p': [1, 2]
                                    },
                            "DecisionTree": {
                                                "criterion": ['gini', 'entropy'],
                                                "max_depth": range(2, 4, 1),
                                                "min_samples_split": [2, 5, 10],
                                                "min_samples_leaf": [1, 5, 10]
                                            },
                            "LogisticRegression": {
                                                    "penalty": ['l1', 'l2'],
                                                    "C": [0.1, 1, 10],
                                                    "max_iter": [500, 1000, 2000]
                                                },
                            "LinearRegression": {"fit_intercept": [True, False]},
                            "AdaBoost": {
                                            "n_estimators": [10, 50, 100, 130],
                                            "learning_rate": [0.1, 0.5, 1, 2]
                                        },
                            "GradientBoosting": {
                                                    "learning_rate": [0.1, 0.5, 1, 2],
                                                    "n_estimators": [10, 50, 100, 130],
                                                    "max_depth": [3, 5, 7, 9],
                                                    "min_samples_split": [2, 5, 10],
                                                    "min_samples_leaf": [1, 5, 10]
                                                },
                            "SVC": {
                                        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                                        'C': [0.1, 1, 10],
                                        'gamma': ['scale', 'auto'],
                                        'degree': [2, 3, 4]
                                    },
                            "GaussianNB": {
                                            'priors': [None],
                                            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
                                        },
                            "XGBoost": {
                                            'learning_rate': [0.5, 0.1, 0.01, 0.001],
                                            'max_depth': [3, 5, 10, 20],
                                            'n_estimators': [10, 50, 100, 200]
                                        }
                            }

            for model_name in self.models.keys():
                
                if model_name in param_grids:
                    model = self.get_best_params_for_model(model_name, param_grids[model_name], train_x, train_y)
                
                else:
                    model = self.models[model_name]
                    model.fit(train_x, train_y)
                
                predictions = model.predict_proba(test_x)
                
                if len(test_y.unique()) == 1:
                    score = accuracy_score(test_y, model.predict(test_x))
                
                else:
                    score = roc_auc_score(test_y, predictions, multi_class='ovr')

                models_scores[model_name] = score
                self.logger_object.log(self.file_object, f'{model_name} Score: {score}')
            
            best_model_name = max(models_scores, key=models_scores.get)
            best_model = self.models[best_model_name]
            self.logger_object.log(self.file_object, f'Best model: {best_model_name} with score: {models_scores[best_model_name]}')

            return best_model_name, best_model

        except Exception as e:
            self.logger_object.log(self.file_object, f'Exception occurred in get_best_model method of the Model_Finder class. Exception message: {str(e)}')
            self.logger_object.log(self.file_object, 'Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            raise Exception()