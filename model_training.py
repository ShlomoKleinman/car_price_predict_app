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

def prepare_data(data): 
    # Drop unnecessary columns
    columns_to_drop = ['Pic_num', 'Cre_date', 'Repub_date', 'Description', 'Test', 'Supply_score', 'Gear', 'Engine_type', 'Prev_ownership', 'Curr_ownership', 'Area', 'City', 'Color']
    data = data.drop(columns=columns_to_drop)  # Drop the unnecessary columns from the DataFrame

    # Replace same name in different language
    data['manufactor'] = data['manufactor'].str.replace('Lexsus', 'לקסוס')  # Replace 'Lexsus' with 'לקסוס' in the 'manufactor' column
    
    # Remove manufactor name from model and keep only main model name
    manufacturers = data['manufactor'].unique()  # Get unique values of 'manufactor'
    def remove_manufacturer_from_model(row):
        model = row['model']
        for manufacturer in manufacturers:
            model = model.replace(manufacturer, '').strip()  # Remove the manufacturer name from the model name
        model = model.split()[0]  # Keep only the main model name
        return model
    data['model'] = data.apply(remove_manufacturer_from_model, axis=1)  # Apply the function to each row in the 'model' column
    
    # Clean and impute capacity_Engine
    try:
        data['capacity_Engine'] = pd.to_numeric(data['capacity_Engine'].str.replace(',', ''), errors='coerce')  # Convert 'capacity_Engine' to numeric, handling commas
    except AttributeError:
        pass
    data['capacity_Engine'] = data.groupby(['manufactor','model','Year'])['capacity_Engine'].transform(lambda x: x.fillna(x.mean()))  # Fill missing values with mean per group
    data['capacity_Engine'] = data.groupby(['manufactor','model'])['capacity_Engine'].transform(lambda x: x.fillna(x.mean()))  # Fill remaining missing values with mean per group
    data['capacity_Engine'] = data.groupby(['manufactor','Year'])['capacity_Engine'].transform(lambda x: x.fillna(x.mean()))  # Fill remaining missing values with mean per group
    
    # Clean and impute Km
    try:
        data['Km'] = pd.to_numeric(data['Km'].str.replace(',', ''), errors='coerce')  # Convert 'Km' to numeric, handling commas
    except AttributeError:
        pass
    data['Km'] = data.groupby(['Year'])['Km'].transform(lambda x: x.fillna(x.mean()))  # Fill missing values with mean per year
    
    # Feature engineering
    current_year = datetime.now().year
    data['km_per_year'] = data['Km'] / (current_year - data['Year'])  # Calculate kilometers per year
    data['age'] = np.maximum(1, current_year - data['Year'])  # Calculate the age of the car
    data['km_age_ratio'] = data['Km'] / data['age']  # Calculate the ratio of kilometers to age
    
    # Log transform of continuous variables
    data['log_km'] = np.log1p(data['Km'])  # Apply log transformation to 'Km'
    data['log_capacity_Engine'] = np.log1p(data['capacity_Engine'])  # Apply log transformation to 'capacity_Engine'
    
    # Seperate In Order To Handle missing values before polynomial features
    numeric_columns = data.select_dtypes(include=[np.number]).columns  # Select numeric columns
    non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns  # Select non-numeric columns
    
    # Impute numeric columns
    numeric_imputer = SimpleImputer(strategy='mean')  # Create a SimpleImputer for numeric columns with mean strategy
    data[numeric_columns] = numeric_imputer.fit_transform(data[numeric_columns])  # Impute missing values in numeric columns
    
    # Impute non-numeric columns
    non_numeric_imputer = SimpleImputer(strategy='most_frequent')  # Create a SimpleImputer for non-numeric columns with most frequent strategy
    data[non_numeric_columns] = non_numeric_imputer.fit_transform(data[non_numeric_columns])  # Impute missing values in non-numeric columns
    
    # One-hot encoding
    categorical_columns = ['manufactor', 'model']
    data = pd.get_dummies(data, columns=[col for col in categorical_columns if col in data.columns])  # One-hot encode categorical columns
    
    # Polynomial features
    required_columns = ['Year', 'Hand', 'capacity_Engine', 'Km', 'km_per_year','age']  # Columns to use for polynomial features
    
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)  # Create PolynomialFeatures transformer
    poly_features = poly.fit_transform(data[required_columns])  # Fit and transform polynomial features
    poly_feature_names = poly.get_feature_names_out(required_columns)  # Get names of polynomial features
    data = data.drop(columns=required_columns)  # Drop original columns
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)  # Create DataFrame with polynomial features
    data = pd.concat([data.reset_index(drop=True), poly_df], axis=1)  # Concatenate original data with polynomial features
    
    # Normalization
    scaler = MinMaxScaler()  # Create MinMaxScaler
    data[required_columns] = scaler.fit_transform(data[required_columns])  # Normalize specified columns
    
    return data


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
