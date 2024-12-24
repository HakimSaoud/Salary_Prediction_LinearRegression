import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.title("Salary Prediction App")
st.write("Predict salaries based on input values using a preloaded dataset.")


df = pd.read_csv("salaries.csv")


x = df.iloc[:, :-1].values
y = df.iloc[:, 1].values


model = LinearRegression()
model.fit(x, y)



st.write("### Predict a Salary")
user_input = st.number_input(
    "Enter a value for prediction (e.g., years of experience):", 
    min_value=int(np.min(x)), 
    max_value=int(np.max(x)), 
    step=1
)

if st.button("Predict"):
    user_input_reshaped = np.array([[user_input]])
    prediction = model.predict(user_input_reshaped)
    st.write(f"### Predicted Salary: {prediction[0]:.2f}")
