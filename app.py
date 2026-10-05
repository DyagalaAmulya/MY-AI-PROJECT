import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------
st.title("🎓 AI-Based Student Performance Prediction System")
st.write(
    "Predict student academic performance using Machine Learning."
)

# ---------------------------------------------------------
# CREATE SAMPLE DATASET
# ---------------------------------------------------------
@st.cache_data
def create_dataset():

    np.random.seed(42)

    n = 500

    data = pd.DataFrame({
        "Study_Hours": np.random.randint(1, 10, n),
        "Attendance": np.random.randint(50, 101, n),
        "Previous_Marks": np.random.randint(35, 101, n),
        "Assignments": np.random.randint(30, 101, n),
        "Sleep_Hours": np.random.randint(4, 10, n),
        "Internet_Usage": np.random.randint(1, 8, n)
    })

    # Calculate performance score
    score = (
        data["Study_Hours"] * 5
        + data["Attendance"] * 0.25
        + data["Previous_Marks"] * 0.35
        + data["Assignments"] * 0.15
        + data["Sleep_Hours"] * 1.5
        - data["Internet_Usage"] * 2
    )

    # Convert score into performance classes
    data["Performance"] = pd.cut(
        score,
        bins=[-np.inf, 70, 100, np.inf],
        labels=["Low", "Average", "High"]
    )

    return data


data = create_dataset()

# ---------------------------------------------------------
# DISPLAY DATASET
# ---------------------------------------------------------
with st.expander("📊 View Training Dataset"):
    st.dataframe(data.head(20))

# ---------------------------------------------------------
# PREPARE DATA
# ---------------------------------------------------------
X = data[
    [
        "Study_Hours",
        "Attendance",
        "Previous_Marks",
        "Assignments",
        "Sleep_Hours",
        "Internet_Usage"
    ]
]

y = data["Performance"]

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# ---------------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------------------------------------------------
# MODEL ACCURACY
# ---------------------------------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

# ---------------------------------------------------------
# SIDEBAR INPUT
# ---------------------------------------------------------
st.sidebar.header("📝 Student Details")

study_hours = st.sidebar.slider(
    "Study Hours per Day",
    min_value=1,
    max_value=12,
    value=5
)

attendance = st.sidebar.slider(
    "Attendance (%)",
    min_value=50,
    max_value=100,
    value=75
)

previous_marks = st.sidebar.slider(
    "Previous Exam Marks (%)",
    min_value=0,
    max_value=100,
    value=60
)

assignments = st.sidebar.slider(
    "Assignment Completion (%)",
    min_value=0,
    max_value=100,
    value=70
)

sleep_hours = st.sidebar.slider(
    "Sleep Hours per Day",
    min_value=1,
    max_value=12,
    value=7
)

internet_usage = st.sidebar.slider(
    "Internet Usage Hours per Day",
    min_value=0,
    max_value=12,
    value=3
)

# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
student_data = pd.DataFrame({
    "Study_Hours": [study_hours],
    "Attendance": [attendance],
    "Previous_Marks": [previous_marks],
    "Assignments": [assignments],
    "Sleep_Hours": [sleep_hours],
    "Internet_Usage": [internet_usage]
})

if st.sidebar.button("🔮 Predict Performance"):

    prediction = model.predict(student_data)
    probabilities = model.predict_proba(student_data)

    predicted_class = label_encoder.inverse_transform(prediction)[0]

    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------
    st.subheader("📌 Prediction Result")

    if predicted_class == "High":
        st.success(
            f"🎉 Predicted Performance: **{predicted_class}**"
        )

    elif predicted_class == "Average":
        st.warning(
            f"📚 Predicted Performance: **{predicted_class}**"
        )

    else:
        st.error(
            f"⚠️ Predicted Performance: **{predicted_class}**"
        )

    # -----------------------------------------------------
    # PROBABILITY
    # -----------------------------------------------------
    st.subheader("📈 Prediction Probability")

    probability_df = pd.DataFrame(
        probabilities,
        columns=label_encoder.classes_
    )

    probability_df = probability_df.T
    probability_df.columns = ["Probability"]

    st.bar_chart(probability_df)

    # -----------------------------------------------------
    # STUDENT DETAILS
    # -----------------------------------------------------
    st.subheader("👨‍🎓 Student Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Study Hours", study_hours)

    with col2:
        st.metric("Attendance", f"{attendance}%")

    with col3:
        st.metric("Previous Marks", f"{previous_marks}%")

    col4, col5, col6 = st.columns(3)

    with col4:
        st.metric("Assignments", f"{assignments}%")

    with col5:
        st.metric("Sleep", f"{sleep_hours} hours")

    with col6:
        st.metric("Internet Usage", f"{internet_usage} hours")

# ---------------------------------------------------------
# MODEL INFORMATION
# ---------------------------------------------------------
st.divider()

st.subheader("🤖 Machine Learning Model")

st.write(
    "The system uses a **Random Forest Classifier** to predict "
    "student performance."
)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Model Accuracy",
        f"{accuracy * 100:.2f}%"
    )

with col2:
    st.metric(
        "Training Records",
        len(X_train)
    )

# ---------------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------------
st.subheader("📊 Feature Importance")

importance = model.feature_importances_

feature_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
}).sort_values(
    by="Importance",
    ascending=False
)

fig, ax = plt.subplots()

ax.bar(
    feature_df["Feature"],
    feature_df["Importance"]
)

ax.set_xlabel("Features")
ax.set_ylabel("Importance")
ax.set_title("Factors Affecting Student Performance")

plt.xticks(rotation=30)

st.pyplot(fig)

# ---------------------------------------------------------
# RECOMMENDATIONS
# ---------------------------------------------------------
st.subheader("💡 AI Recommendations")

recommendations = []

if study_hours < 4:
    recommendations.append(
        "Increase daily study time."
    )

if attendance < 75:
    recommendations.append(
        "Improve class attendance."
    )

if previous_marks < 50:
    recommendations.append(
        "Focus on improving fundamental concepts."
    )

if assignments < 60:
    recommendations.append(
        "Complete assignments regularly."
    )

if sleep_hours < 6:
    recommendations.append(
        "Maintain sufficient sleep for better concentration."
    )

if internet_usage > 6:
    recommendations.append(
        "Reduce unnecessary internet usage during study time."
    )

if len(recommendations) == 0:
    recommendations.append(
        "Great! Maintain your current study habits."
    )

for recommendation in recommendations:
    st.info("✅ " + recommendation)

# ---------------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------------
joblib.dump(model, "student_performance_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

st.divider()

st.caption(
    "AI-Based Student Performance Prediction System | "
    "Python + Streamlit + Scikit-learn"
)