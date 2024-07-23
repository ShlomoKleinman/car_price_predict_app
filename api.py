from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
from car_data_prep import prepare_data
import traceback
from datetime import datetime

# Create a Flask application instance
app = Flask(__name__)

# Load the trained model from a file
model = joblib.load('trained_model.pkl')

# Load your dataset from a CSV file
data = pd.read_csv("dataset.csv", encoding='utf-8')

# Replace manufacturer names in different languages to a standard form
data['manufactor'] = data['manufactor'].str.replace('Lexsus', 'לקסוס')

# Function to remove manufacturer name from model names to have a unique list inside the app
manufacturers = data['manufactor'].unique()
def remove_manufacturer_from_model(row):
    model = row['model']
    for manufacturer in manufacturers:
        model = model.replace(manufacturer, '').strip()
    model = model.split()[0]  # Keep only the main model name
    return model

data['model'] = data.apply(remove_manufacturer_from_model, axis=1)

# Extract unique values for drop-down menus in the app
unique_manufacturers = data['manufactor'].unique()
unique_models = data['model'].unique()
unique_engine_volumes = data['capacity_Engine'].unique()

# Create a dictionary of manufacturers and their respective models
manufacturer_models = {}
for manufacturer in unique_manufacturers:
    manufacturer_models[manufacturer] = data[data['manufactor'] == manufacturer]['model'].unique().tolist()

# Define the home page route
@app.route('/')
@app.route('/index')
def index():
    name = 'אורח'  # Default user name
    current_year = datetime.now().year  # Get the current year
    return render_template('index.html', 
                           title='ברוכים הבאים', 
                           username=name,
                           manufacturers=unique_manufacturers,
                           models=unique_models,
                           engine_volumes=unique_engine_volumes,
                           manufacturer_models=manufacturer_models,
                           selected_year=current_year,
                           current_year=current_year)

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        year = int(request.form['year'])
        manufacturer = request.form['manufacturer']
        model_name = request.form['model']
        hand = int(request.form['hand'])
        engine_volume = float(request.form['engine_volume'])
        mileage = float(request.form['mileage'])

        # Verify that the model belongs to the selected manufacturer
        if model_name not in manufacturer_models.get(manufacturer, []):
            return render_template('index.html', error="Invalid model for selected manufacturer")

        # Log the received data
        print(f"Received data: year={year}, manufacturer={manufacturer}, model={model_name}, "
              f"hand={hand}, engine_volume={engine_volume}, mileage={mileage}")

        # Prepare data for prediction
        input_data = {
            'Year': [year],
            'manufactor': [manufacturer],
            'model': [model_name],
            'Hand': [hand],
            'capacity_Engine': [engine_volume],
            'Km': [mileage]
        }
        input_df = pd.DataFrame(input_data)

        # Log the input DataFrame
        print("Input DataFrame:")
        print(input_df)

        # Use the prepare_data function to preprocess the data
        processed_input = prepare_data(input_df)

        # Log the processed input
        print("Processed input:")
        print(processed_input)
        print("Processed input shape:", processed_input.shape)

        # Ensure all columns present in the training data are also in the test data
        for col in model.feature_names_in_:
            if col not in processed_input.columns:
                processed_input[col] = 0

        # Remove any extra columns that are not present during training
        processed_input = processed_input[model.feature_names_in_]

        # Log the final input for prediction
        print("Final input for prediction:")
        print(processed_input)
        print("Final input shape:", processed_input.shape)

        # Make prediction
        predicted_price = model.predict(processed_input)[0]

        # Log the raw predicted price
        print(f"Raw predicted price: {predicted_price}")

        # Format predicted price without decimals and with commas
        formatted_price = '{:,.0f}'.format(predicted_price)

        # Render the result page with the prediction
        current_year = datetime.now().year
        return render_template(
            'index.html',
            title='תוצאת החיזוי',
            username='אורח',
            price=formatted_price,
            selected_year=year,
            manufacturer=manufacturer,
            model=model_name,
            hand=hand,
            engine_volume=engine_volume,
            mileage=mileage,
            manufacturers=unique_manufacturers,
            models=unique_models,
            engine_volumes=unique_engine_volumes,
            current_year=current_year,
            manufacturer_models=manufacturer_models
        )

    except Exception as e:
        # Log the error and return an error response
        print(f"An error occurred: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 400

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
