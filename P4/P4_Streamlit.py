import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.title("🎬 IMDB Sentiment Analysis")

# Create two columns
col1, col2 = st.columns(2)

# Load dataset
try:
    df = pd.read_csv("IMDB_Dataset.csv")
    df['sentiment'] = df['sentiment'].map({'positive':1, 'negative':0})
    
    # Left column: Dataset and Model Info
    with col1:
        st.subheader("📊 Dataset Overview")
        st.write(f"**Total Reviews:** {len(df)}")
        st.write(f"**Positive Reviews:** {sum(df['sentiment'] == 1)}")
        st.write(f"**Negative Reviews:** {sum(df['sentiment'] == 0)}")
        
        # Show sample data
        with st.expander("👀 View Sample Data"):
            st.dataframe(df.head())
        
        st.markdown("---")
        st.subheader("⚙️ Model Configuration")
        
        max_features = st.slider("Max Features", 1000, 10000, 5000, step=1000)
        test_size = st.slider("Test Size (%)", 10, 40, 20)
        
        if st.button("🚀 Train Model", key="train", use_container_width=True):
            with st.spinner("Training Naive Bayes model..."):
                # Prepare data
                X = df['review']
                y = df['sentiment']
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size/100, random_state=42, stratify=y
                )
                
                # Vectorize text
                tfidf = TfidfVectorizer(stop_words="english", max_features=max_features)
                X_train_tfidf = tfidf.fit_transform(X_train)
                X_test_tfidf = tfidf.transform(X_test)
                
                # Train model
                nb_model = MultinomialNB()
                nb_model.fit(X_train_tfidf, y_train)
                y_pred = nb_model.predict(X_test_tfidf)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)
                
                # Store in session state
                st.session_state['tfidf'] = tfidf
                st.session_state['model'] = nb_model
                st.session_state['accuracy'] = accuracy
                st.session_state['report'] = report
                st.session_state['cm'] = cm
                
                st.success("✅ Model trained successfully!")
                st.metric("Model Accuracy", f"{accuracy:.2%}")

except FileNotFoundError:
    with col1:
        st.error("❌ File 'IMDB_Dataset.csv' not found!")
        st.info("Please ensure the CSV file is in the same directory")

# Right column: Results and Prediction
with col2:
    st.subheader("📈 Results")
    
    if 'accuracy' in st.session_state:
        # Show classification report
        with st.expander("📋 Classification Report"):
            st.text(st.session_state['report'])
        
        # Show confusion matrix
        st.write("### Confusion Matrix")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(st.session_state['cm'], annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Negative", "Positive"],
                    yticklabels=["Negative", "Positive"],
                    ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Prediction section
    st.subheader("🔮 Predict Sentiment")
    
    # Text area for review input
    review_text = st.text_area(
        "Enter your movie review:",
        height=150,
        placeholder="Type or paste a movie review here..."
    )
    
    if st.button("📝 Analyze Sentiment", key="predict", use_container_width=True):
        if 'model' not in st.session_state:
            st.warning("⚠️ Please train the model first!")
        elif not review_text.strip():
            st.warning("⚠️ Please enter a review text!")
        else:
            # Get model and vectorizer from session state
            tfidf = st.session_state['tfidf']
            model = st.session_state['model']
            
            # Vectorize and predict
            review_tfidf = tfidf.transform([review_text])
            prediction = model.predict(review_tfidf)
            
            # Display result
            result = "Positive" if prediction[0] == 1 else "Negative"
            
            if result == "Positive":
                st.success(f"😊 **Sentiment:** {result}")
            else:
                st.error(f"😞 **Sentiment:** {result}")
            
            # Show accuracy
            if 'accuracy' in st.session_state:
                st.info(f"Model accuracy: {st.session_state['accuracy']:.2%}")
    else:
        st.info("Enter a movie review and click 'Analyze Sentiment' to predict")

# Show info if no model trained yet
if 'accuracy' not in st.session_state:
    with col2:
        st.info("👈 Configure settings and train the model first")