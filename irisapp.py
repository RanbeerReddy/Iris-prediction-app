import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
target_names = iris.target_names

# Cross-validation
model = DecisionTreeClassifier(random_state=42)
cv_scores = cross_val_score(model, X, y, cv=5)

# Train-test split for confusion matrix, report, and prediction
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation metrics
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=target_names)
conf_matrix = confusion_matrix(y_test, y_pred)

# Streamlit UI
st.set_page_config(page_title="Iris Classifier", layout="centered")
st.title("🌼 Iris Flower Classifier")
st.markdown("Use the sliders to input features and predict the species.")

# Feature input form
with st.form("iris_form"):
    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.slider("Sepal Length (cm)", float(X.iloc[:, 0].min()), float(X.iloc[:, 0].max()), 5.1)
        sepal_width = st.slider("Sepal Width (cm)", float(X.iloc[:, 1].min()), float(X.iloc[:, 1].max()), 3.5)
    with col2:
        petal_length = st.slider("Petal Length (cm)", float(X.iloc[:, 2].min()), float(X.iloc[:, 2].max()), 1.4)
        petal_width = st.slider("Petal Width (cm)", float(X.iloc[:, 3].min()), float(X.iloc[:, 3].max()), 0.2)
    
    submitted = st.form_submit_button("Predict")

# Prediction
if submitted:
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]
    predicted_species = target_names[prediction].capitalize()
    st.success(f"🌸 Predicted Species: **{predicted_species}**")

# Evaluation
st.subheader("📊 Model Evaluation")

# Show test accuracy
st.write(f"**Test Accuracy:** {acc:.2f}")

# Show K-Fold CV scores
st.markdown("**K-Fold Cross-Validation Accuracy (5 folds):**")
st.write(f"Scores: {cv_scores}")
st.write(f"Mean CV Accuracy: {cv_scores.mean():.2f}")

# Classification report
st.markdown("**Classification Report:**")
st.text(report)

# Confusion matrix
st.markdown("**Confusion Matrix:**")
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)
