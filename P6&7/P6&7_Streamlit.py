import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

st.title("🚢 Titanic Survival Predictor")

# Create two columns
col1, col2 = st.columns(2)

# Left column: File upload and Data Processing
with col1:
    st.subheader("📁 Upload Dataset")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose Titanic Dataset CSV file",
        type=['csv'],
        help="Upload Titanic dataset CSV file"
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            
            st.success("✅ File uploaded successfully!")
            
            # Show dataset info
            with st.expander("📊 Dataset Overview"):
                st.write(f"**Total Passengers:** {len(df)}")
                st.write(f"**Survived:** {sum(df.get('Survived', 0) == 1)}")
                st.write(f"**Did Not Survive:** {sum(df.get('Survived', 0) == 0)}")
                
                # Show first few rows
                st.write("**First 5 rows:**")
                st.dataframe(df.head())
            
            st.markdown("---")
            st.subheader("⚙️ Data Processing")
            
            # Check for required columns
            required_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked', 'Survived']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing columns: {', '.join(missing_cols)}")
                st.stop()
            
            # Show data processing steps
            st.write("**Data Processing Steps:**")
            
            # 1. Handle missing values
            if st.checkbox("Handle Missing Values", value=True):
                df['Age'] = df['Age'].fillna(df['Age'].median())
                df['Fare'] = df['Fare'].fillna(df['Fare'].median())
                df = df.dropna(subset=['Embarked'])
                st.write("✓ Missing values handled")
            
            # 2. Convert categorical variables
            if st.checkbox("Convert Categorical Variables", value=True):
                df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
                st.write("✓ Categorical variables converted")
            
            # Prepare features and target
            features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
            X = df[features]
            y = df['Survived']
            
            # Show processed data
            with st.expander("👀 View Processed Data"):
                st.write(f"**Features used:** {', '.join(features)}")
                st.dataframe(X.head())
            
            st.markdown("---")
            st.subheader("🔧 Model Training")
            
            test_size = st.slider("Test Size (%)", 10, 40, 20)
            
            if st.button("🚀 Train Model", key="train", use_container_width=True):
                with st.spinner("Training Logistic Regression model..."):
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size/100, random_state=42
                    )
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Train model
                    model = LogisticRegression(max_iter=1000)
                    model.fit(X_train_scaled, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test_scaled)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    report = classification_report(y_test, y_pred)
                    cm = confusion_matrix(y_test, y_pred)
                    
                    # Store in session state
                    st.session_state['df'] = df
                    st.session_state['model'] = model
                    st.session_state['scaler'] = scaler
                    st.session_state['accuracy'] = accuracy
                    st.session_state['report'] = report
                    st.session_state['cm'] = cm
                    st.session_state['features'] = features
                    
                    st.success("✅ Model trained successfully!")
                    st.metric("Model Accuracy", f"{accuracy:.3f}")
                    
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    else:
        st.info("📤 Please upload a Titanic dataset CSV file to begin")

# Right column: Results and Prediction
with col2:
    st.subheader("📈 Model Results")
    
    if 'accuracy' in st.session_state:
        # Show classification report
        with st.expander("📋 Classification Report"):
            st.text(st.session_state['report'])
        
        # Show confusion matrix
        st.write("### Confusion Matrix")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(st.session_state['cm'], annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Prediction section
    st.subheader("🔮 Predict Survival")
    
    if 'features' in st.session_state:
        # Create input form for prediction
        with st.form("prediction_form"):
            st.write("Enter passenger details:")
            
            # Input fields
            col_a, col_b = st.columns(2)
            
            with col_a:
                pclass = st.selectbox("Passenger Class", [1, 2, 3], help="1 = 1st, 2 = 2nd, 3 = 3rd")
                age = st.number_input("Age", min_value=0, max_value=100, value=30)
                sibsp = st.number_input("Siblings/Spouses", min_value=0, max_value=10, value=0)
                parch = st.number_input("Parents/Children", min_value=0, max_value=10, value=0)
            
            with col_b:
                fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=50.0)
                sex = st.selectbox("Sex", ["Male", "Female"])
                embarked = st.selectbox("Embarked", ["S", "C", "Q"], help="S = Southampton, C = Cherbourg, Q = Queenstown")
            
            # Submit button
            submitted = st.form_submit_button("🎯 Predict Survival", use_container_width=True)
            
            if submitted:
                # Prepare input data
                input_data = {
                    'Pclass': pclass,
                    'Age': age,
                    'SibSp': sibsp,
                    'Parch': parch,
                    'Fare': fare,
                    'Sex_male': 1 if sex == "Male" else 0,
                    'Embarked_Q': 1 if embarked == "Q" else 0,
                    'Embarked_S': 1 if embarked == "S" else 0
                }
                
                # Convert to DataFrame and scale
                input_df = pd.DataFrame([input_data])
                input_scaled = st.session_state['scaler'].transform(input_df)
                
                # Make prediction
                prediction = st.session_state['model'].predict(input_scaled)[0]
                probability = st.session_state['model'].predict_proba(input_scaled)[0]
                
                # Display result
                if prediction == 1:
                    st.success(f"✅ **Prediction: SURVIVED**")
                    st.write(f"Survival probability: {probability[1]:.1%}")
                else:
                    st.error(f"❌ **Prediction: DID NOT SURVIVE**")
                    st.write(f"Non-survival probability: {probability[0]:.1%}")
                
                # Show accuracy
                if 'accuracy' in st.session_state:
                    st.info(f"Model accuracy: {st.session_state['accuracy']:.3f}")
    else:
        st.info("👈 Upload a dataset and train the model first")

# Sidebar with example download
with st.sidebar:
    st.markdown("---")
    st.subheader("📥 Need Titanic data?")
    
    # Example Titanic data
    example_data = {
        'PassengerId': [1, 2, 3],
        'Survived': [0, 1, 1],
        'Pclass': [3, 1, 3],
        'Name': ['Braund, Mr. Owen Harris', 'Cumings, Mrs. John Bradley', 'Heikkinen, Miss. Laina'],
        'Sex': ['male', 'female', 'female'],
        'Age': [22.0, 38.0, 26.0],
        'SibSp': [1, 1, 0],
        'Parch': [0, 0, 0],
        'Ticket': ['A/5 21171', 'PC 17599', 'STON/O2. 3101282'],
        'Fare': [7.25, 71.28, 7.92],
        'Embarked': ['S', 'C', 'S']
    }
    
    example_df = pd.DataFrame(example_data)
    csv = example_df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="Download Example Data",
        data=csv,
        file_name="titanic_example.csv",
        mime="text/csv",
        help="Download sample Titanic data"
    )
    
    st.markdown("""
    **Required columns:**
    - Pclass
    - Age
    - SibSp
    - Parch
    - Fare
    - Sex
    - Embarked
    - Survived
    """)