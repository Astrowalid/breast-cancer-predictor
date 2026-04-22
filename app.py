import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ==========================================
# 1. PAGE SETUP & TITLES
# ==========================================
st.set_page_config(page_title="Breast Cancer Predictor", page_icon="🩺", layout="centered")

st.title("🩺 Breast Cancer Prediction Tool")
st.write("This application uses a Machine Learning model to predict whether a tumor is **Malignant** or **Benign** based on cell nucleus measurements.")
st.write("*(Note: This is an academic project, not a real medical diagnostic tool.)*")
st.markdown("---")

# ==========================================
# 2. LOAD THE BRAIN AND THE TRANSLATOR
# ==========================================
# We use @st.cache_resource so the app only loads these files once, making it fast.
@st.cache_resource
def load_models():
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_models()

# ==========================================
# 3. USER INPUT (SIDEBAR) - UPDATED TO TOP 5
# ==========================================
st.sidebar.header("Patient Measurements")
st.sidebar.write("Adjust the sliders for the top 5 most important features:")

# I have updated the sliders to match your chart, with reasonable min/max values
texture_worst = st.sidebar.slider("Worst Texture", min_value=12.0, max_value=50.0, value=25.0, step=0.1)
radius_worst = st.sidebar.slider("Worst Radius", min_value=7.0, max_value=40.0, value=16.0, step=0.1)
concave_points_worst = st.sidebar.slider("Worst Concave Points", min_value=0.0, max_value=0.3, value=0.11, step=0.001)
perimeter_worst = st.sidebar.slider("Worst Perimeter", min_value=50.0, max_value=255.0, value=107.0, step=1.0)
area_worst = st.sidebar.slider("Worst Area", min_value=180.0, max_value=4250.0, value=880.0, step=10.0)

# ==========================================
# 4. DATA FORMATTING (INDEX MAPPED)
# ==========================================
def prepare_input_data(scaler):
    # 1. Start with the average baseline for ALL 30 features
    input_array = scaler.mean_.copy()
    
    # 2. Overwrite ONLY our Top 5 features at their exact specific indices!
    input_array[20] = radius_worst
    input_array[21] = texture_worst
    input_array[22] = perimeter_worst
    input_array[23] = area_worst
    input_array[27] = concave_points_worst
    
    # 3. Reshape for the model
    return input_array.reshape(1, -1)

# ==========================================
# 5. PREDICTION LOGIC
# ==========================================
if st.button("Run Prediction", type="primary"):
    
    # 1. Get the raw numbers
    raw_data = prepare_input_data(scaler)
    
    # 2. Scale the numbers using our saved scaler
    scaled_data = scaler.transform(raw_data)
    
    # 3. Make the prediction (0 = Malignant, 1 = Benign based on our earlier encoding)
    prediction = model.predict(scaled_data)[0]
    
    # 4. Get the confidence probability (if the model supports it)
    try:
        probability = model.predict_proba(scaled_data)[0]
        confidence = max(probability) * 100
    except AttributeError:
        # Some models like standard SVM don't output probabilities easily
        confidence = None

    # ==========================================
    # 6. DISPLAY RESULTS
    # ==========================================
    st.markdown("### Diagnosis Result:")
    
    if prediction == 0:
        st.error(f"🚨 **Prediction: MALIGNANT**")
    else:
        st.success(f"✅ **Prediction: BENIGN**")
        
    if confidence:
        st.write(f"**Model Confidence:** {confidence:.2f}%")

# ==========================================
# 7. PROJECT CONTRIBUTORS
# ==========================================
st.markdown("---") # Draws a neat horizontal line to separate the app from the footer
st.subheader("👨‍💻 Project Contributors")
st.write("This machine learning application was developed by:")

# Create two equal-sized columns
col1, col2 = st.columns(2)

with col1:
    st.info("""
    **Walid IDBENNACER** | [LinkedIn](https://www.linkedin.com/in/walid-idbennacer-65b42a215/)
    """)
    
with col2:
    st.info("""
    **Chamss-Eddine ERRABEH** | [LinkedIn](https://www.linkedin.com/in/chamss-eddine-errabeh-915610272/)
    """)
