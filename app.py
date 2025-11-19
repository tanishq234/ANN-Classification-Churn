import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

model = None
onehot_encoder_geo = None
label_encoder_gender = None
scaler = None

# Load preprocessing artifacts first (we need scaler to infer input size if the HDF5 contains weights only)
try:
    with open('onehot_encoder_geo.pkl','rb') as file:
        onehot_encoder_geo = pickle.load(file)
    with open('label_encoder_gender.pkl','rb') as file:
        label_encoder_gender = pickle.load(file)
    with open('scaler.pkl','rb') as file:
        scaler = pickle.load(file)
    load_pickle_error = None
except Exception as e:
    load_pickle_error = e

# Inspect model.h5: it may contain full model (config + weights) or only weights.
# If it's only weights (common if saved with model.save_weights), we need to rebuild the architecture and load weights.
import h5py, os
model_file = 'model.h5'
if os.path.exists(model_file):
    try:
        with h5py.File(model_file, 'r') as f:
            h5_keys = list(f.keys())
    except Exception:
        h5_keys = []
else:
    h5_keys = []

load_model_error = None
if 'model_config' in h5_keys:
    # file contains full model config — try regular load
    try:
        model = tf.keras.models.load_model(model_file, compile=False)
    except Exception as e:
        model = None
        load_model_error = e
else:
    # file likely contains only weights (e.g., keys like 'model_weights') — rebuild architecture and load weights
    try:
        # determine input size from scaler if available
        if scaler is not None and hasattr(scaler, 'feature_names_in_'):
            input_dim = len(scaler.feature_names_in_)
        else:
            # fallback: try loading saved feature_order.pkl
            try:
                import pickle as _p
                feature_order = _p.load(open('feature_order.pkl', 'rb'))
                input_dim = len(feature_order)
            except Exception:
                input_dim = None

        if input_dim is None:
            raise RuntimeError('Cannot infer model input size: scaler.feature_names_in_ not available and feature_order.pkl not found.')

        # Recreate the model architecture used during training (update this if you changed it)
        from tensorflow.keras.models import Sequential as _Sequential
        from tensorflow.keras.layers import Dense as _Dense

        model = _Sequential([
            _Dense(64, activation='relu', input_shape=(input_dim,)),
            _Dense(32, activation='relu'),
            _Dense(1, activation='sigmoid')
        ])

        # load weights
        model.load_weights(model_file)
        load_model_error = None
    except Exception as e:
        model = None
        load_model_error = e


#streamlit app

st.title('Customer Churn prediction')


geography=st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('credit Score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number of products',1,4)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('is Active Member',[0,1])


#Prepare input data

input_data=pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [num_of_products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember' : [is_active_member],
    'EstimatedSalary' : [estimated_salary]
})

#One-hot encoded 'Geography'

geo_encoded=onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df= pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))


input_data=pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# If loading failed earlier, show an error and stop the app instead of crashing
if load_model_error is not None:
    st.error(f"Failed to load Keras model: {load_model_error}")
    st.info("Common causes: model saved with a different TensorFlow/Keras version, or model.h5 is corrupted. Try re-saving the model with `model.save('model.h5')` using your current TF version.")
    st.stop()

if load_pickle_error is not None:
    st.error(f"Failed to load preprocessing artifacts (scaler/encoders): {load_pickle_error}")
    st.info("Make sure 'scaler.pkl', 'onehot_encoder_geo.pkl', and 'label_encoder_gender.pkl' exist and are readable from the app working directory.")
    st.stop()

# All artifacts loaded; proceed to scale
input_data_scaled = scaler.transform(input_data)


#Predict churn

prediction=model.predict(input_data_scaled)
prediction_proba=prediction[0][0]

st.write(f'Churn Probablity: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to Churn')

else:
    st.write('The customer is not likely to Churn') 
