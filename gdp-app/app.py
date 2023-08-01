from flask import Flask, request, jsonify

from model import *

app = Flask(__name__)

filename = 'gdp_model.sav'
model = joblib.load(filename)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        continent = data['Continent']
        subject = data['Subject Descriptor']
        values = data['values']

        sample_input = pd.DataFrame({'Subject Descriptor':subject,"2023":values, 'Continent': continent })
        sample_input_encoded = ct.transform(sample_input)
        prediction = model.predict(sample_input_encoded)

        predicted_gdp = prediction[0, 0]

        return jsonify({'predicted_gdp_as_at_2023': predicted_gdp})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
