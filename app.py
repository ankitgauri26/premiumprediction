import streamlit as st
import pickle
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin
# from sklearn.preprocessing import StandardScaler

# Load the trained model
# with open('optimized_xgb_pipeline.pkl', 'rb') as file:
#     model = pickle.load(file)

# model = joblib.load("optimized_xgb_pipeline.pkl")

# Load the scaler used for standardization
# with open('scaler_updated.pkl', 'rb') as file:
#     scaler = pickle.load(file)

# Load the scaler
scaler_loaded = joblib.load("standardScaler.pkl")



# Define a function to preprocess inputs
# def preprocess_input(weight, height, age, num_surgeries, binary_features):
#     bmi = weight / ((height*0.01) ** 2)
#     numeric_features = pd.DataFrame({
#         'Age': [age],
#         'BMI': [bmi],
#         'NumberOfMajorSurgeries': [num_surgeries]
#     })
#     standardized_numeric_features = scaler.transform(numeric_features)
#     input_data = pd.DataFrame(standardized_numeric_features, columns=['Age', 'BMI', 'NumberOfMajorSurgeries'])
#     for feature in binary_features:
#         input_data[feature] = [st.session_state[feature]]
#     return input_data

@st.cache_resource
def load_model():
    model = joblib.load("optimized_xgb_pipeline_fixed.pkl")

# Define the custom wrapper class
class LogToOriginalXGBRegressor(BaseEstimator, RegressorMixin):
    def _init_(self, base_model=None):
        self.base_model = base_model if base_model else xgb.XGBRegressor(objective="reg:squarederror", random_state=42)

    def fit(self, X, y):
        self.base_model.fit(X, y)
        return self

    def predict(self, X):
        log_preds = self.base_model.predict(X)
        return np.expm1(log_preds)  # Convert back to original scale


model = joblib.load("optimized_xgb_pipeline_fixed.pkl")

def preprocess_and_predict(raw_data, label_encoders):
    required_columns = ["age", "gender", "bmi", "smoker", "region", "medical_history",
                        "family_medical_history", "exercise_frequency", "occupation", "coverage_level"]
    raw_data = raw_data[required_columns]

    # raw_data['gender'] = raw_data['gender'].map({'male': 1, 'female': 0})
    # raw_data['smoker'] = raw_data['smoker'].map({'yes': 1, 'no': 0})

    # st.write(raw_data)
    # for col in ["gender", "smoker"]:
    #     if col in label_encoders:
    #         raw_data[col] = label_encoders[col].transform(raw_data[col])

    # categorical_cols = ["region", "medical_history", "family_medical_history",
    #                     "exercise_frequency", "occupation", "coverage_level"]
    # raw_data = pd.get_dummies(raw_data, columns=categorical_cols)
    # st.write(raw_data)  
    # for col in one_hot_columns:
    #     if col not in raw_data.columns:
    #         raw_data[col] = 0

    # raw_data = raw_data[scaler.feature_names_in_]
    # scaler = StandardScaler().fit(raw_data)
    # st.write("This is for testing")
    # st.write(scaler_loaded)
    # scaled_data = scaler_loaded.transform(raw_data)
    # st.write(scaled_data)
    # model = joblib.load("optimized_xgb_pipeline.pkl")
    # load_model()
    # st.write("Before loading")
    # feature_names = model.named_steps['preprocessor'].get_feature_names_out()
    # st.write(feature_names)
    # st.write(raw_data)
    # model = joblib.load("optimized_xgb_pipeline_fixed.pkl")
    # st.write("after loading")
    # st.write(model)
    premium_predict = model.predict(raw_data)
    return premium_predict
    # charges_pred = np.exp(log_charges_pred)
    # raw_data["Predicted_Premium"] = charges_pred
    # st.write(raw_data)
    # return raw_data

# Streamlit app
st.title('Premium Prediction App')

# Collect user inputs
age = st.number_input('Age', min_value=18, max_value=65, format='%d')
gender = st.radio("Gender", ["male", "female"], index=0, horizontal=True)
bmi = st.number_input('BMI', min_value=21.0, format='%f')
smoker = st.radio("Is Smoker", ["yes", "no"], index=1, horizontal=True)
region = st.selectbox("Region", ("northeast", "northwest", "southeast", "southwest"))
medical_history = st.selectbox("Medical History", ["Diabetes", "No Disease", "High blood pressure", "Heart disease"])
family_medical_history = st.selectbox("Family Medical History", ["Diabetes", "No Disease", "High blood pressure", "Heart disease"])
exercise_frequency = st.selectbox("Excercise Frequency", ["Never","Occasionally", "Rarely", "Frequently"])
occupation = st.selectbox("Occupation", ["Blue collar", "white collar", "Student", "Unemployed"])
coverage_level = st.selectbox("Coverage Level", ["Basic", "Standard", "Premium"])


if st.button("Calculate Premium") :
    #st.write("You selected:", {'age': age, 'gender': gender, 'bmi': bmi, 'smoker': smoker, 'region': region, 'medical_history': medical_history, 'family_medical_history': family_medical_history, 'exercise_frequency': exercise_frequency, 'occupation': occupation, 'coverage_level': coverage_level})
    prediction = preprocess_and_predict(pd.DataFrame({'age': age, 'gender': gender, 'bmi': bmi, 'smoker': smoker, 'region': region, 'medical_history': medical_history, 'family_medical_history': family_medical_history, 'exercise_frequency': exercise_frequency, 'occupation': occupation, 'coverage_level': coverage_level}, index=[0]), ["region", "smoker"],)
    st.write(f'The predicted premium is: {prediction[0]:.2f}')