# 1. Import Necessary Libraries
from flask import Flask, request, render_template_string
import pickle
import pandas as pd
import numpy as np
import warnings

# Ignore warnings for a cleaner output
warnings.filterwarnings('ignore')

# 2. Initialize Flask App
app = Flask(__name__)

# 3. Load the Trained Model
try:
    # This must match the filename of the model from your training notebook
    model = pickle.load(open('best_productivity_model.pkl', 'rb'))
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print("❌ Error: 'best_productivity_model.pkl' not found.")
    print("Please ensure the model file from your notebook is in the correct directory.")
    model = None
except Exception as e:
    print(f"❌ An error occurred while loading the model: {e}")
    model = None

# 4. Define Mappings for Categorical Features
# This mimics the LabelEncoder used in the training notebook.
# The numeric values are based on alphabetical sorting, which is how LabelEncoder works.
QUARTER_MAP = {'Quarter1': 0, 'Quarter2': 1, 'Quarter3': 2, 'Quarter4': 3, 'Quarter5': 4}
DEPARTMENT_MAP = {'finishing': 0, 'sewing': 1}
DAY_MAP = {'Monday': 0, 'Saturday': 1, 'Sunday': 2, 'Thursday': 3, 'Tuesday': 4, 'Wednesday': 5}


# --- HTML TEMPLATES ---
# Using render_template_string to keep everything in one file.
# Styled with Tailwind CSS for a modern look.

HOME_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Employee Productivity Prediction</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        body { font-family: 'Inter', sans-serif; }
    </style>
</head>
<body class="bg-gray-100 text-gray-800">
    <div class="container mx-auto p-4 md:p-8 max-w-4xl">
        <div class="bg-white rounded-2xl shadow-lg p-8">
            <h1 class="text-3xl md:text-4xl font-bold text-center text-gray-900 mb-2">Employee Productivity Predictor</h1>
            <p class="text-center text-gray-500 mb-8">Enter the employee's details to predict their productivity level.</p>

            <form action="/" method="post">
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    
                    <!-- Categorical Features -->
                    <div>
                        <label for="quarter" class="block text-sm font-medium text-gray-700">Quarter</label>
                        <select name="quarter" required class="mt-1 block w-full bg-gray-50 border border-gray-300 rounded-lg shadow-sm p-3 focus:ring-indigo-500 focus:border-indigo-500">
                            <option value="Quarter1">Quarter 1</option>
                            <option value="Quarter2">Quarter 2</option>
                            <option value="Quarter3">Quarter 3</option>
                            <option value="Quarter4">Quarter 4</option>
                            <option value="Quarter5">Quarter 5</option>
                        </select>
                    </div>
                    <div>
                        <label for="department" class="block text-sm font-medium text-gray-700">Department</label>
                        <select name="department" required class="mt-1 block w-full bg-gray-50 border border-gray-300 rounded-lg shadow-sm p-3 focus:ring-indigo-500 focus:border-indigo-500">
                            <option value="sewing">Sewing</option>
                            <option value="finishing">Finishing</option>
                        </select>
                    </div>
                    <div>
                        <label for="day" class="block text-sm font-medium text-gray-700">Day of the Week</label>
                        <select name="day" required class="mt-1 block w-full bg-gray-50 border border-gray-300 rounded-lg shadow-sm p-3 focus:ring-indigo-500 focus:border-indigo-500">
                            <option value="Monday">Monday</option>
                            <option value="Tuesday">Tuesday</option>
                            <option value="Wednesday">Wednesday</option>
                            <option value="Thursday">Thursday</option>
                            <option value="Saturday">Saturday</option>
                            <option value="Sunday">Sunday</option>
                        </select>
                    </div>
                     <div>
                        <label for="month" class="block text-sm font-medium text-gray-700">Month</label>
                        <input type="number" name="month" required class="mt-1 block w-full bg-gray-50 border border-gray-300 rounded-lg shadow-sm p-3 focus:ring-indigo-500 focus:border-indigo-500" placeholder="e.g., 1 for January">
                    </div>

                    <!-- Core Metrics -->
                    <div>
                        <label for="team" class="block text-sm font-medium text-gray-700">Team Number</label>
                        <input type="number" name="team" required class="mt-1 block w-full bg-gray-50 border border-gray-300 rounded-lg shadow-sm p-3 focus:ring-indigo-500 focus:border-indigo-500" placeholder="e.g., 8">
                    </div>
                    <div>
                        <label for="targeted_productivity" class="block text-sm font-medium text-gray-700">Targeted Productivity</label>
                        <input type="number" step="0.01" name="targeted_productivity" required class="mt-1 block w-full bg-gray-50 border border-gray-300 rounded-lg shadow-sm p-3 focus:ring-indigo-500 focus:border-indigo-500" placeholder="e.g., 0.80">
                    </div>
                    <div>
                        <label for="smv" class="block text-sm font-medium text-gray-700">Standard Minute Value (SMV)</label>
                        <input type="number" step="0.01" name="smv" required class="mt-1 block w-full bg-gray-50 border border-gray-300 rounded-lg shadow-sm p-3 focus:ring-indigo-500 focus:border-indigo-500" placeholder="e.g., 26.16">
                    </div>
                    <div>
                        <label for="no_of_workers" class="block text-sm font-medium text-gray-700">Number of Workers</label>
                        <input type="number" step="0.1" name="no_of_workers" required class="mt-1 block w-full bg-gray-50 border border-gray-300 rounded-lg shadow-sm p-3 focus:ring-indigo-500 focus:border-indigo-500" placeholder="e.g., 59.0">
                    </div>

                    <!-- Additional Metrics -->
                    <div class="md:col-span-2 grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div>
                            <label for="over_time" class="block text-sm font-medium text-gray-700">Over Time (mins)</label>
                            <input type="number" name="over_time" required class="mt-1 block w-full bg-gray-50 border border-gray-300 rounded-lg shadow-sm p-3" placeholder="e.g., 7080">
                        </div>
                        <div>
                            <label for="incentive" class="block text-sm font-medium text-gray-700">Incentive (BDT)</label>
                            <input type="number" name="incentive" required class="mt-1 block w-full bg-gray-50 border border-gray-300 rounded-lg shadow-sm p-3" placeholder="e.g., 98">
                        </div>
                        <div>
                            <label for="idle_time" class="block text-sm font-medium text-gray-700">Idle Time (hours)</label>
                            <input type="number" step="0.1" name="idle_time" required class="mt-1 block w-full bg-gray-50 border border-gray-300 rounded-lg shadow-sm p-3" value="0">
                        </div>
                        <div>
                            <label for="idle_men" class="block text-sm font-medium text-gray-700">Idle Men</label>
                            <input type="number" name="idle_men" required class="mt-1 block w-full bg-gray-50 border border-gray-300 rounded-lg shadow-sm p-3" value="0">
                        </div>
                         <div>
                            <label for="no_of_style_change" class="block text-sm font-medium text-gray-700">Style Changes</label>
                            <input type="number" name="no_of_style_change" required class="mt-1 block w-full bg-gray-50 border border-gray-300 rounded-lg shadow-sm p-3" value="0">
                        </div>
                    </div>
                </div>

                <div class="mt-8">
                    <button type="submit" class="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-3 px-4 rounded-lg shadow-md transition duration-300 ease-in-out transform hover:scale-105">
                        Predict Productivity
                    </button>
                </div>
            </form>
            
            <!-- Prediction Result Area -->
            {% if prediction_text %}
            <div class="mt-10 text-center p-6 rounded-lg 
                {% if 'Highly' in prediction_text %} bg-green-100 border-green-500 
                {% elif 'Medium' in prediction_text %} bg-yellow-100 border-yellow-500
                {% else %} bg-red-100 border-red-500 {% endif %} border-l-4">
                <h2 class="text-2xl font-semibold 
                    {% if 'Highly' in prediction_text %} text-green-800 
                    {% elif 'Medium' in prediction_text %} text-yellow-800
                    {% else %} text-red-800 {% endif %}">
                    Prediction Result
                </h2>
                <p class="text-lg mt-2 
                    {% if 'Highly' in prediction_text %} text-green-700 
                    {% elif 'Medium' in prediction_text %} text-yellow-700
                    {% else %} text-red-700 {% endif %}">
                    {{ prediction_text }} (Actual Predicted Score: {{ score }})
                </p>
            </div>
            {% endif %}

        </div>
    </div>
</body>
</html>
"""

# 5. Define Flask Routes
@app.route("/", methods=['GET', 'POST'])
def predict():
    prediction_text = ""
    score_text = ""
    if request.method == 'POST':
        if model is None:
            return "Model not loaded. Please check server logs.", 500

        try:
            # --- Data Collection and Transformation ---
            # Get raw data from the form
            quarter_str = request.form['quarter']
            department_str = request.form['department']
            day_str = request.form['day']
            
            # Map string values to the integer codes the model expects
            quarter = QUARTER_MAP.get(quarter_str, 0)
            department = DEPARTMENT_MAP.get(department_str, 0)
            day = DAY_MAP.get(day_str, 0)

            # Get the rest of the numerical data
            team = int(request.form['team'])
            targeted_productivity = float(request.form['targeted_productivity'])
            smv = float(request.form['smv'])
            over_time = int(request.form['over_time'])
            incentive = int(request.form['incentive'])
            idle_time = float(request.form['idle_time'])
            idle_men = int(request.form['idle_men'])
            no_of_style_change = int(request.form['no_of_style_change'])
            no_of_workers = float(request.form['no_of_workers'])
            month = int(request.form['month'])

            # --- Create Input Array for Prediction ---
            # The order of features MUST match the training data from the notebook
            input_features = np.array([[
                quarter, department, day, team, targeted_productivity, smv,
                over_time, incentive, idle_time, idle_men, no_of_style_change,
                no_of_workers, month
            ]])

            # --- Prediction ---
            prediction = model.predict(input_features)
            pred_value = round(float(prediction[0]), 3)
            score_text = str(pred_value)

            # --- Categorize Prediction ---
            if pred_value >= 0.75:
                prediction_text = 'The employee is Highly Productive.'
            elif pred_value >= 0.5:
                prediction_text = 'The employee has Medium Productivity.'
            else:
                prediction_text = 'The employee has Low Productivity.'

        except Exception as e:
            print(f"Error during prediction: {e}")
            prediction_text = f"Error: Could not process request. Details: {e}"

    # Render the page
    return render_template_string(HOME_TEMPLATE, prediction_text=prediction_text, score=score_text)


# 6. Run the App
if __name__ == '__main__':
    app.run(debug=True, port=5001)
