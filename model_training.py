import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectFromModel
from scipy.stats import uniform
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from datetime import datetime
from car_data_prep import prepare_data
import warnings
from sklearn.exceptions import ConvergenceWarning

# Ignore ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

data = pd.read_csv('dataset.csv')

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler


prepared_data = prepare_data(data)


# splitting data
X = prepared_data.drop(columns=['Price'])
y = prepared_data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Hyperparameter tuning using Grid Search 
param_grid = {
    'alpha': [0.005,0.01,0.05, 0.1],
    'l1_ratio': [0.9,0.93,0.94, 0.95, 0.97,0.99]
    #'max_iter': [1000, 2000, 5000,10000]
}
elastic_net = ElasticNet(random_state=42)
grid_search = GridSearchCV(elastic_net, param_grid, cv=10, scoring='neg_mean_squared_error',verbose=1)
grid_search.fit(X_train, y_train)

# Best model from Grid Search
best_model = grid_search.best_estimator_


# Print best parameters
print("Best parameters:", best_model)

# Feature importance
feature_importance = np.abs(best_model.coef_)  # Calculate the absolute values of the coefficients from the model
feature_names = X.columns  # Get the names of the features from the DataFrame

# Pair the feature importances with their corresponding feature names and sort them in descending order
important_5features = sorted(zip(feature_importance, feature_names), reverse=True)[:5]  # Get the top 5 features

print('Top 5 Features:')
for importance, name in important_5features:
    # Determine if the impact is positive or negative based on the original coefficient value
    sign = 'Positive' if best_model.coef_[list(feature_names).index(name)] > 0 else 'Negative'
    # Print the feature name, its impact sign, and the coefficient value
    print(f'{name}: {sign} impact with coefficient {importance}')

# Perform 10-fold cross-validation on the entire dataset
cv_scores = cross_val_score(best_model, X, y, cv=10, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores)
print("Cross-validation RMSE scores:", cv_rmse)
print("Mean CV RMSE:", cv_rmse.mean())
print("Standard deviation of CV RMSE:", cv_rmse.std())

# # Save the final model
import joblib
joblib.dump(best_model, 'final_model.pkl')
