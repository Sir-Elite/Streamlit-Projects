import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Simple Iris Flower Prediction App

This app predicts the flower type of a flower!
""")

st.sidebar.header('User Input Parameters')


def user_input_features():
    # st.sidebar.slider('title', min, max, default)
    sepal_length = st.sidebar.slider('Sepal Length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal Width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal Length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal Width', 0.1, 2.5, 0.2)

    data = {
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }

    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()

st.subheader('User Input Parameters')
st.write(df)

iris = datasets.load_iris()
x = iris.data
y = iris.target

classifier = RandomForestClassifier()
classifier.fit(x, y)

prediction = classifier.predict(df)
prediction_probability = classifier.predict_proba(df)

st.subheader('Class Labels and Their Corresponding Index Number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])

st.subheader('Prediction Probability')
st.write(prediction_probability)