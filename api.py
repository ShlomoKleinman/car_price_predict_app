from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
from car_data_prep import prepare_data
import traceback
from datetime import datetime

app = Flask(__name__)

# Load the trained model
model = joblib.load('trained_model.pkl')

# Load your dataset
data = pd.read_csv("dataset.csv", encoding='utf-8')

# Replace same name in different language
data['manufactor'] = data['manufactor'].str.replace('Lexsus', 'לקסוס')

# Remove manufactor name from model and keep only main model name in order to have unique list inside the app
manufacturers = data['manufactor'].unique()
def remove_manufacturer_from_model(row):
        model = row['model']
        for manufacturer in manufacturers:
            model = model.replace(manufacturer, '').strip()
        model = model.split()[0]
        return model
data['model'] = data.apply(remove_manufacturer_from_model, axis=1)

# Extract unique values in order to have unique list inside the app (user friendly)
unique_manufacturers = data['manufactor'].unique() + ["אחר"]
unique_models = data['model'].unique() + ["אחר"]
unique_engine_volumes = data['capacity_Engine'].unique()

manufacturer_models = {}
for manufacturer in unique_manufacturers:
    manufacturer_models[manufacturer] = data[data['manufactor'] == manufacturer]['model'].unique().tolist()

@app.route('/')
@app.route('/index')
def index():
    name = 'אורח'
    current_year = datetime.now().year
    return render_template('index.html', 
                           title='ברוכים הבאים', 
                           username=name,
                           manufacturers=unique_manufacturers,
                           models=unique_models,
                           engine_volumes=unique_engine_volumes,
                           manufacturer_models=manufacturer_models,
                           selected_year=current_year,
                           current_year=current_year)

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

        # Verify that the model belongs to the manufacturer
        if model_name not in manufacturer_models.get(manufacturer, []):
            return render_template('index.html', error="Invalid model for selected manufacturer")

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
        print("Input DataFrame:")
        print(input_df)

        # Use the prepare_data function
        processed_input = prepare_data(input_df)
        
        print("Processed input:")
        print(processed_input)
        print("Processed input shape:", processed_input.shape)

        # Ensure all columns present in the training data are also in the test data
        for col in model.feature_names_in_:
            if col not in processed_input.columns:
                processed_input[col] = 0

        # Remove any extra columns that are not present during training
        processed_input = processed_input[model.feature_names_in_]

        print("Final input for prediction:")
        print(processed_input)
        print("Final input shape:", processed_input.shape)

        # Make prediction
        predicted_price = model.predict(processed_input)[0]

        print(f"Raw predicted price: {predicted_price}")

        # Format predicted price without decimals and with commas
        formatted_price = '{:,.0f}'.format(predicted_price)

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
        print(f"An error occurred: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
