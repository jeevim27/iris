
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st 


iris = load_iris()
X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)
svm_classifier.fit(X_train, y_train)


st.title("Iris Flower Classification App")
st.sidebar.header("Input Parameters")

sepal_length = st.sidebar.slider("Sepal Length", 4.0, 8.0, 5.0)
sepal_width = st.sidebar.slider("Sepal Width", 2.0, 4.5, 3.5)
petal_length = st.sidebar.slider("Petal Length", 1.0, 7.0, 1.0)
petal_width = st.sidebar.slider("Petal Width", 0.1, 2.5, 0.1)


input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
input_data = scaler.transform(input_data)  
prediction = svm_classifier.predict(input_data)
predicted_class = iris.target_names[prediction[0]]

st.write("Predicted Class:", predicted_class)
