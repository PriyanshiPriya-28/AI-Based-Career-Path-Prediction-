from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load artifacts
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
model_columns = pickle.load(open("columns.pkl", "rb"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/process')
def process():
    return render_template('process.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get form data
        form_data = request.form.to_dict()
        
        # 2. Prepare the input DataFrame
        # We start with numerical fields
        input_dict = {
            'GPA': [float(form_data.get('GPA', 0))],
            'Extracurricular_Activities': [int(form_data.get('Extracurricular_Activities', 0))],
            'Internships': [int(form_data.get('Internships', 0))],
            'Projects': [int(form_data.get('Projects', 0))],
            'Leadership_Positions': [int(form_data.get('Leadership_Positions', 0))],
            'Field_Specific_Courses': [int(form_data.get('Field_Specific_Courses', 0))],
            'Research_Experience': [int(form_data.get('Research_Experience', 0))],
            'Coding_Skills': [int(form_data.get('Coding_Skills', 0))],
            'Communication_Skills': [int(form_data.get('Communication_Skills', 0))],
            'Problem_Solving_Skills': [int(form_data.get('Problem_Solving_Skills', 0))],
            'Teamwork_Skills': [int(form_data.get('Teamwork_Skills', 0))],
            'Analytical_Skills': [int(form_data.get('Analytical_Skills', 0))],
            'Presentation_Skills': [int(form_data.get('Presentation_Skills', 0))],
            'Networking_Skills': [int(form_data.get('Networking_Skills', 0))],
            'Industry_Certifications': [int(form_data.get('Industry_Certifications', 0))]
        }
        
        # 3. Handle 'Field' dummy variables
        selected_field = form_data.get('Field')
        # Initialize all field dummy columns to 0
        for col in model_columns:
            if col.startswith('Field_'):
                input_dict[col] = [1 if col == f"Field_{selected_field}" else 0]

        input_df = pd.DataFrame(input_dict)
        
        # Ensure column order matches training
        input_df = input_df[model_columns]
        
        # 4. Scale and Predict Probabilities
        final_features = scaler.transform(input_df)
        probabilities = model.predict_proba(final_features)[0]
        
        # 5. Get Top 10 Predictions
        classes = model.classes_
        top_10_idx = np.argsort(probabilities)[-10:][::-1]
        
        top_10_predictions = []
        for idx in top_10_idx:
            top_10_predictions.append({
                'career': classes[idx],
                'confidence': round(probabilities[idx] * 100, 2)
            })

        return render_template('result.html', predictions=top_10_predictions)

    except Exception as e:
        return f"Error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True, port=5000)