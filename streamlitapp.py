import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


@st.cache_data
def load_data():
    df = pd.read_csv("heart.csv") 
    return df

df = load_data()
st.title("❤️ Heart Disease Prediction Tool")
st.write("Dataset Preview:")
st.dataframe(df.head())


if st.checkbox("Show EDA"):
    st.subheader("Data Info")
    st.write(df.describe())
    st.write("Missing values:", df.isnull().sum().sum())

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

X = df.drop("target", axis=1)  
y = df["target"]               

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.subheader("Model Performance")
st.write(f"✅ Accuracy: {acc:.2f}")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

st.subheader("Confusion Matrix")
fig2, ax2 = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax2)
st.pyplot(fig2)


st.sidebar.header("🔮 Predict Heart Disease")
def user_input():
    age = st.sidebar.slider("Age", 20, 80, 40)
    sex = st.sidebar.selectbox("Sex", (0,1))  
    cp = st.sidebar.slider("Chest Pain Type (0-3)", 0, 3, 1)
    trestbps = st.sidebar.slider("Resting Blood Pressure", 90, 200, 120)
    chol = st.sidebar.slider("Cholesterol", 100, 400, 200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = true; 0 = false)", (0,1))
    restecg = st.sidebar.slider("Resting ECG (0-2)", 0, 2, 1)
    thalach = st.sidebar.slider("Max Heart Rate Achieved", 70, 220, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina (1 = yes; 0 = no)", (0,1))
    oldpeak = st.sidebar.slider("ST depression induced", 0.0, 6.0, 1.0)
    slope = st.sidebar.slider("Slope (0-2)", 0, 2, 1)
    ca = st.sidebar.slider("Major vessels colored (0-4)", 0, 4, 0)
    thal = st.sidebar.slider("Thal (0 = normal; 1 = fixed defect; 2 = reversible defect)", 0, 2, 1)

    data = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
        "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input()
st.subheader("User Input Features")
st.write(input_df)

input_scaled = scaler.transform(input_df)

prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

st.subheader("Prediction Result")
st.write("🔴 Heart Disease Detected" if prediction[0]==1 else "🟢 No Heart Disease")
st.write("Prediction Probability:", prediction_proba)
