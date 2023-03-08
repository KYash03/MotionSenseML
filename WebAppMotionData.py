import streamlit as st
import joblib
import pandas as pd

st.title('Height, Weight, Age, Gender and Activity Predictor')

numerical_pipeline = joblib.load('final_pipeline_numerical')
categorical_pipeline = joblib.load('final_pipeline_categorical')

st.sidebar.header('Input Parameters')


def get_input():
    data = {}
    parameters = ['attitude.roll', 'attitude.pitch', 'attitude.yaw', 'gravity.x',
                  'gravity.y', 'gravity.z', 'rotationRate.x', 'rotationRate.y',
                  'rotationRate.z', 'userAcceleration.x', 'userAcceleration.y',
                  'userAcceleration.z']
    for parameter in parameters:
        data[parameter] = st.sidebar.number_input(
            parameter, step=1e-6, format="%.6f")
    return data


data = get_input()
user_input_df = pd.DataFrame(data, index=[0])

st.write('Input feature values in the sidebar.')

if sum(data.values()):
    numerical_predictions = numerical_pipeline.predict(user_input_df)[0]
    st.write('Prediction:')
    st.markdown('1. **Height**: {:.2f}'.format(numerical_predictions[1]))
    st.markdown('2. **Weight**: {:.2f}'.format(numerical_predictions[0]))
    st.markdown('3. **Age**: {}'.format(round(numerical_predictions[2])))

    categorical_predictions = categorical_pipeline.predict(user_input_df)[0]
    activities = ['Downstairs', 'Jogging',
                  'Sitting', 'Standing', 'Upstairs', 'Walking']
    gender = ['Female', 'Male']
    st.write('4. **Gender**: {}'.format(gender[categorical_predictions[0]]))
    st.write(
        '6. **Activity**: {}'.format(activities[categorical_predictions[1]]))

st.markdown('***')
st.header('About')
st.write("Hi! Here, you can input data from a specific instance and get accurate predictions of the person's physical attributes and the activity in which they were engaged. The app uses two machine learning models: KNeighborsRegressor for predicting height, weight and age, and KNeighborsClassifier for predicting gender and activity. I hope you like it.")
