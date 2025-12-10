ML Model Hub (Lite)

A simple Streamlit web app that lets users upload their own dataset and instantly run multiple machine-learning models — no coding required.

This project auto-handles preprocessing, trains ML models, evaluates them, and provides clear explanations of the results. Perfect for students, beginners, and quick ML experimentation.

📌 Features

Upload any CSV dataset

Auto preprocessing:

Removes rows with missing target values

Encodes categorical columns

Fills missing feature values

Choose between Classification and Regression models:

Logistic Regression

Random Forest Classifier

Linear Regression

Random Forest Regressor

Clean evaluation metrics:

Accuracy

MSE

RMSE

R² Score

Automatic explanation of results

Smooth beginner-friendly interface built using Streamlit

🛠️ Tech Stack
Component	Technology
UI	Streamlit
ML Models	Scikit-learn
Data Handling	Pandas, NumPy
Language	Python
📥 Installation
1️⃣ Clone the repository
git clone https://github.com/rafanriyaz/ml-model-hub_lite.git
cd ml-model-hub_lite

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the application
streamlit run app.py

📂 Project Structure
ml-model-hub_lite/
│── app.py
│── README.md
│── requirements.txt
└── sample_datasets/   (optional)

🧠 How It Works
1. Upload Dataset

You upload a CSV file.
The app previews the first five rows so you can confirm everything looks correct.

2. Select Target Column

You choose which column you want to predict.

3. Pick a Machine Learning Model

You choose one of the available ML algorithms depending on your task.

4. Training + Evaluation

The app automatically:

Encodes categorical data

Splits data into train/test sets

Trains your chosen model

Calculates performance metrics

5. Explanation

The app gives you a human-friendly interpretation of:

Accuracy

MSE

RMSE

R²

Whether your data is imbalanced

