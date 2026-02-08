import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.title("📈 Diabetes Progression Prediction")

# Load dataset
diabetes = load_diabetes()
x = diabetes.data
y = diabetes.target

# Dataset info
st.write(f"**Dataset:** {x.shape[0]} samples, {x.shape[1]} features")
st.write(f"**Target range:** {y.min():.1f} to {y.max():.1f}")

# Model training controls
st.subheader("Model Training")
col1, col2 = st.columns(2)
with col1:
    test_size = st.slider("Test Size (%)", 10, 40, 20)
with col2:
    if st.button("🚀 Train Model", use_container_width=True):
        with st.spinner("Training Linear Regression model..."):
            # Split data
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=test_size/100, random_state=42
            )
            
            # Train model
            model = LinearRegression()
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store in session state
            st.session_state['x_test'] = x_test
            st.session_state['y_test'] = y_test
            st.session_state['y_pred'] = y_pred
            st.session_state['mse'] = mse
            st.session_state['r2'] = r2
            st.session_state['feature_names'] = diabetes.feature_names

# Show metrics if model trained
if 'mse' in st.session_state:
    st.subheader("Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mean Squared Error", f"{st.session_state['mse']:.2f}")
    with col2:
        st.metric("R-squared Score", f"{st.session_state['r2']:.2f}")
    
    st.markdown("---")
    
    # Larger visualizations
    st.subheader("Model Visualizations")
    
    # Create larger figure for first plot
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.scatter(st.session_state['y_test'], st.session_state['y_pred'], 
                color="blue", alpha=0.6, s=50)
    ax1.plot([st.session_state['y_test'].min(), st.session_state['y_test'].max()],
             [st.session_state['y_test'].min(), st.session_state['y_test'].max()],
             "r--", lw=2)
    ax1.set_title("True vs Predicted Values", fontsize=16)
    ax1.set_xlabel("True Values", fontsize=12)
    ax1.set_ylabel("Predicted Values", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    st.pyplot(fig1)
    
    # Create larger figure for second plot
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.scatter(st.session_state['x_test'][:, 2], st.session_state['y_pred'],
                color="green", alpha=0.7, s=50)
    ax2.set_title("Feature (BMI) vs Predicted Values", fontsize=16)
    ax2.set_xlabel("BMI (Feature 2)", fontsize=12)
    ax2.set_ylabel("Predicted Diabetes Progression", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    st.pyplot(fig2)
    
    # Feature information
    with st.expander("📋 Dataset Features Information"):
        for i, name in enumerate(st.session_state['feature_names']):
            st.write(f"**{i}. {name}**")
            if i == 2:
                st.caption("(BMI is used in the second visualization)")
else:
    st.info("👆 Click the 'Train Model' button to start training and see visualizations")