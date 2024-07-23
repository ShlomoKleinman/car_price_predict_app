import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from datetime import datetime

def prepare_data(data): 
    current_year = datetime.now().year

    # Replace same name in different language
    data['manufactor'] = data['manufactor'].str.replace('Lexsus', 'לקסוס')

    # Remove manufactor name from model and keep only main model name
    manufacturers = data['manufactor'].unique()
    def remove_manufacturer_from_model(row):
        model = row['model']
        for manufacturer in manufacturers:
            model = model.replace(manufacturer, '').strip()
        model = model.split()[0]
        return model
    data['model'] = data.apply(remove_manufacturer_from_model, axis=1)

    # Clean and impute capacity_Engine
    try:
        data['capacity_Engine'] = pd.to_numeric(data['capacity_Engine'].str.replace(',', ''), errors='coerce')
    except AttributeError:
        pass
    data['capacity_Engine'] = data.groupby(['manufactor','model','Year'])['capacity_Engine'].transform(lambda x: x.fillna(x.mean()))
    data['capacity_Engine'] = data.groupby(['manufactor','model'])['capacity_Engine'].transform(lambda x: x.fillna(x.mean()))
    data['capacity_Engine'] = data.groupby(['manufactor','Year'])['capacity_Engine'].transform(lambda x: x.fillna(x.mean()))

    # Clean and impute Km
    try:
        data['Km'] = pd.to_numeric(data['Km'].str.replace(',', ''), errors='coerce')
    except AttributeError:
        pass
    data['Km'] = data.groupby(['Year'])['Km'].transform(lambda x: x.fillna(x.mean()))

    # Feature engineering
    data['age'] = np.maximum(0, current_year - data['Year'])
    data['age'] = np.where(data['age'] == 0, 0.5, data['age'])  # Set age to 0.5 for current year cars
    data['km_per_year'] = np.where(data['age'] > 0, data['Km'] / data['age'], data['Km'])
    data['km_age_ratio'] = np.where(data['age'] > 0, data['Km'] / data['age'], data['Km'])

    # Log transform of continuous variables
    data['log_km'] = np.log1p(data['Km'])
    data['log_capacity_Engine'] = np.log1p(data['capacity_Engine'])

    # Handle infinite values
    data = data.replace([np.inf, -np.inf], np.finfo(np.float64).max)

    # Separate In Order To Handle missing values before polynomial features
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns

    # Impute numeric columns
    numeric_imputer = SimpleImputer(strategy='mean')
    data[numeric_columns] = numeric_imputer.fit_transform(data[numeric_columns])

    # Impute non-numeric columns
    non_numeric_imputer = SimpleImputer(strategy='most_frequent')
    data[non_numeric_columns] = non_numeric_imputer.fit_transform(data[non_numeric_columns])

    # One-hot encoding
    categorical_columns = ['manufactor', 'model']
    data = pd.get_dummies(data, columns=[col for col in categorical_columns if col in data.columns])

    # Polynomial features
    required_columns = ['Year', 'Hand', 'capacity_Engine', 'Km', 'km_per_year','age']
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly_features = poly.fit_transform(data[required_columns])
    poly_feature_names = poly.get_feature_names_out(required_columns)
    data = data.drop(columns=required_columns)
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)
    data = pd.concat([data.reset_index(drop=True), poly_df], axis=1)

    # Normalization
    scaler = MinMaxScaler()
    data[required_columns] = scaler.fit_transform(data[required_columns])

    # Final check for infinite values
    data = data.replace([np.inf, -np.inf], np.finfo(np.float64).max)

    return data


