import pandas as pd
import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

model = pickle.load(open("model.pkl", "rb"))

df = pd.read_csv("diabetes.csv")

st.sidebar.title("🩺 Diabetes App")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "",
    ["🏠 Home", "🔍 Prediction", "📋 Data View"]
)

if page == "🏠 Home":

    st.title("🩺 Diabetes Prediction System")
    st.subheader("Machine Learning Based Health Risk Assessment")

    st.write(
        "This application predicts whether a patient is likely to have diabetes based on medical information."
    )

    st.markdown("### 📌 What this app does")

    st.write("🔍 Prediction → Predict diabetes risk")
    st.write("📋 Data View → View the dataset")

    st.markdown("### 📊 Dataset Information")

    col1, col2, col3 = st.columns(3)

    diabetic = int(df["Outcome"].sum())

    with col1:
        st.metric("Total Patients", len(df))

    with col2:
        st.metric("Diabetic Patients", diabetic)

    with col3:
        st.metric("Model Accuracy", "74%")
    
    st.markdown("### ⚙️ How It Works")

    st.write("Step 1 📋 Enter patient details")
    st.write("Step 2 🤖 Random Forest model analyzes the data")
    st.write("Step 3 ✅ Receive diabetes prediction")

    st.markdown("### 💡 Problem")

    st.write(
        "Early detection of diabetes is important for preventing serious health complications."
    )

    st.markdown("### 🤖 Solution")

    st.write(
        "This machine learning model predicts whether a patient is likely to have diabetes using medical attributes."
    )
    
    st.markdown("### 📊 Diabetes Distribution")

    diabetic_count = df["Outcome"].sum()
    non_diabetic_count = len(df) - diabetic_count

    fig, ax = plt.subplots()

    ax.pie(
        [diabetic_count, non_diabetic_count],
        labels=["Diabetic", "Non-Diabetic"],
        autopct="%1.1f%%"
    )

    st.pyplot(fig)

    st.markdown("### 🔑 Features Used")

    col1, col2 = st.columns(2)

    with col1:
        st.write("✅ Pregnancies")
        st.write("✅ Glucose")
        st.write("✅ Blood Pressure")
        st.write("✅ Skin Thickness")

    with col2:
        st.write("✅ Insulin")
        st.write("✅ BMI")
        st.write("✅ Diabetes Pedigree Function")
        st.write("✅ Age")

elif page == "🔍 Prediction":

    st.title("🩺 Diabetes Prediction System")
    st.subheader("Machine Learning Based Health Risk Assessment")

    st.write("Enter patient details below:")

    preg = st.number_input("Pregnancies", min_value=0)
    glucose = st.number_input("Glucose", min_value=0)
    bp = st.number_input("Blood Pressure", min_value=0)
    skin = st.number_input("Skin Thickness", min_value=0)
    insulin = st.number_input("Insulin", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
    age = st.number_input("Age", min_value=0)

    if st.button("Predict"):

        data = np.array([
            preg,
            glucose,
            bp,
            skin,
            insulin,
            bmi,
            dpf,
            age
        ]).reshape(1, -1)

        prediction = model.predict(data)

        if prediction[0] == 1:
            st.error("Patient is likely Diabetic")
        else:
            st.success("Patient is likely Not Diabetic")

elif page == "📋 Data View":

    st.title("📋 Diabetes Dataset")

    st.write("Dataset used for training the machine learning model.")

    st.dataframe(df)

    st.write("Dataset Shape:", df.shape)

    st.write("Columns:", list(df.columns))
