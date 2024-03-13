from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('decision_tree_model.joblib')

# Mapping of choices to numerical values
choice_mapping = {
    'BAT': 0,
    'BSBA': 1,
    'BSCS': 2,
    'BSHM': 3,
    'CTE': 4
}

# Mapping of numerical predictions to course names
course_mapping = {
    'BAT': 'Bachelor of Agricultural Technology (BAT)',
    'BSBA': 'Bachelor of Science in Business Administration (BSBA)',
    'BSCS': 'Bachelor of Science in Computer Science (BSCS)',
    'BSHM': 'Bachelor of Science in Hospitality Management (BSHM)',
    'CTE': 'College of Teacher Education (CTE)'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Get user input from the form
    choice_1 = request.form['choice_1']
    choice_2 = request.form['choice_2']
    english = float(request.form['english'])
    math = float(request.form['math'])
    science = float(request.form['science'])
    abstract_reasoning = float(request.form['abstract_reasoning'])

    # Convert choices to numerical values
    choice_1_numeric = choice_mapping.get(choice_1, -1)
    choice_2_numeric = choice_mapping.get(choice_2, -1)

    # Calculate the total
    total = english + math + science + abstract_reasoning

    # order the inputs
    input_features = [[choice_1_numeric, choice_2_numeric, english, math, science, abstract_reasoning, total]]

    # Make predictions
    numerical_prediction = model.predict(input_features)[0]

    # Convert numerical prediction to course name
    predicted_course = course_mapping.get(numerical_prediction, 'Cannot recommend course, try again.')

    # Pass the results to the template
    return render_template('index.html', total=total, prediction=predicted_course)

if __name__ == '__main__':
    app.run(debug=True)