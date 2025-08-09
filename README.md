Employee Productivity Prediction
This project is a machine learning application designed to predict the productivity of employees in the garment industry based on various work-related attributes. The application is built with a Flask backend and provides a simple web interface for users to input data and receive a productivity prediction.

🚀 Features
Web-Based Interface: An intuitive and stylish user interface for entering employee data.

Machine Learning Model: Utilizes a trained Random Forest or XGBoost model to make accurate predictions.

Real-Time Prediction: Instantly provides a productivity score and categorizes it as "High," "Medium," or "Low."

Deployed Application: The project is deployed and accessible online via Render.

🛠️ Tech Stack
Backend: Python, Flask

Machine Learning: Scikit-learn, Pandas, NumPy, XGBoost

Deployment: Render, Gunicorn

Frontend: HTML, Tailwind CSS

📂 File Structure
The repository contains the following files:

app.py: The main Flask application file that contains the backend logic and serves the web interface.

best_productivity_model.pkl: The pre-trained machine learning model saved as a pickle file.

requirements.txt: A list of all the Python dependencies required to run the project.

Employee_Perf_Prediction.ipynb: The Jupyter Notebook used for data exploration, model training, and evaluation.

garments_worker_productivity.csv: The dataset used to train the model.

🖥️ How to Run Locally
To run this project on your local machine, follow these steps:

1. Clone the Repository
git clone [https://github.com/your-username/Employee_Performance_Pred.git](https://github.com/your-username/Employee_Performance_Pred.git)
cd Employee_Performance_Pred

2. Create a Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. Install Dependencies
Make sure you have all the required libraries by running:

pip install -r requirements.txt

4. Run the Flask Application
Start the local server by running the app.py file:

python app.py

The application will be available at http://127.0.0.1:5001 in your web browser.

🌐 Deployment
This application is deployed on Render and can be accessed at the following URL:

employee-pred-byabhay.onrender.com

📋 Usage
Navigate to the application's URL.

Fill in all the fields in the form with the employee's data.
