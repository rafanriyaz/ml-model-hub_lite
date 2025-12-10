import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
import numpy as np

def explain_results(model_type, score, y_test=None, preds=None):
    explanation = ""

    if model_type == "classification":
        accuracy = score
        explanation += f"**Accuracy = {accuracy:.4f}**\n\n"
        explanation += f"- Your model correctly predicted **{accuracy*100:.2f}%** of the samples.\n"

        
        if y_test is not None:
            counts = np.bincount(y_test)
            if max(counts) / sum(counts) > 0.8:
                explanation += "\n**Warning:** Your dataset is imbalanced. Accuracy may be misleading.\n"

    elif model_type == "regression":
        mse = score
        rmse = np.sqrt(mse)
        explanation += f"**MSE = {mse:.4f}**\n"
        explanation += f"**RMSE = {rmse:.4f}**\n\n"
        explanation += f"- RMSE means your predictions are off by **~{rmse:.2f} units** on average.\n"

        
        if y_test is not None and preds is not None:
            ss_res = np.sum((y_test - preds)**2)
            ss_tot = np.sum((y_test - np.mean(y_test))**2)
            r2 = 1 - ss_res / ss_tot
            explanation += f"\n**R² Score = {r2:.4f}**\n"
            explanation += "- R² near 1 means strong prediction quality.\n"
            explanation += "- R² near 0 means weak prediction quality.\n"

    return explanation

st.title(" ML Model Hub")

uploaded_file = st.file_uploader("Upload a CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of uploaded dataset")
    st.dataframe(df.head())

    target = st.selectbox("Select target column", df.columns)

    model_choice = st.selectbox(
        "Choose a Model",
        [
            "Logistic Regression (classification)",
            "Random Forest Classifier",
            "Linear Regression (regression)",
            "Random Forest Regressor"
        ]
    )

    if st.button("Run Model"):
        df_processed = df.copy()

        df_processed = df_processed.dropna(subset=[target])

        for col in df_processed.columns:
            if df_processed[col].dtype == "object":
                df_processed[col] = LabelEncoder().fit_transform(df_processed[col])

        X = df_processed.drop(columns=[target])
        y = df_processed[target]

        X = X.fillna(X.mode().iloc[0])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        if "Logistic Regression" in model_choice:
            model = LogisticRegression(max_iter=500)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = accuracy_score(y_test, preds)

            st.success(f"Accuracy: {score}")
            st.markdown("###  Explanation")
            st.markdown(explain_results("classification", score, y_test, preds))

        elif "Random Forest Classifier" in model_choice:
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = accuracy_score(y_test, preds)

            st.success(f"Accuracy: {score}")
            st.markdown("###  Explanation")
            st.markdown(explain_results("classification", score, y_test, preds))

        elif "Linear Regression" in model_choice:
            model = LinearRegression()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mse = mean_squared_error(y_test, preds)

            st.success(f"MSE: {mse}")
            st.markdown("###  Explanation")
            st.markdown(explain_results("regression", mse, y_test, preds))

        else:
            model = RandomForestRegressor()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mse = mean_squared_error(y_test, preds)

            st.success(f"MSE: {mse} RMSE: {np.sqrt(mse)}")
            st.markdown("###  Explanation")
            st.markdown(explain_results("regression", mse, y_test, preds))