import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import io

st.title("🎬 IMDB Sentiment Analysis")

# Create two columns
col1, col2 = st.columns(2)

# Left column: File upload and Model Info
with col1:
    st.subheader("📁 Upload Dataset")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose IMDB Dataset CSV file",
        type=['csv'],
        help="Upload a CSV file with 'review' and 'sentiment' columns"
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            
            # Check if required columns exist
            if 'review' not in df.columns or 'sentiment' not in df.columns:
                st.error("❌ CSV file must contain 'review' and 'sentiment' columns")
                st.stop()
            
            # Map sentiment values
            df['sentiment'] = df['sentiment'].map({'positive':1, 'negative':0})
            
            # Show dataset info
            st.success("✅ File uploaded successfully!")
            st.write(f"**Total Reviews:** {len(df)}")
            st.write(f"**Positive Reviews:** {sum(df['sentiment'] == 1)}")
            st.write(f"**Negative Reviews:** {sum(df['sentiment'] == 0)}")
            
            # Show sample data
            with st.expander("👀 View Sample Data"):
                st.dataframe(df.head())
            
            st.markdown("---")
            st.subheader("⚙️ Model Configuration")
            
            # Model configuration
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
                    st.session_state['df'] = df
                    st.session_state['tfidf'] = tfidf
                    st.session_state['model'] = nb_model
                    st.session_state['accuracy'] = accuracy
                    st.session_state['report'] = report
                    st.session_state['cm'] = cm
                    
                    st.success("✅ Model trained successfully!")
                    st.metric("Model Accuracy", f"{accuracy:.2%}")
                    
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
    else:
        st.info("📤 Please upload a CSV file to begin")
        st.markdown("""
        **Expected CSV format:**
        - Column 1: `review` (text reviews)
        - Column 2: `sentiment` ('positive' or 'negative')
        """)

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
    
    # Example reviews for quick testing
    with st.expander("💡 Try these examples"):
        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            if st.button("Positive Example", use_container_width=True):
                st.session_state['example_review'] = "This movie was absolutely fantastic! The acting was superb and the storyline kept me engaged from start to finish."
        with col_ex2:
            if st.button("Negative Example", use_container_width=True):
                st.session_state['example_review'] = "Terrible movie. Poor acting, boring plot, and a complete waste of time. Would not recommend to anyone."
    
    # Use example if set
    if 'example_review' in st.session_state:
        review_text = st.session_state['example_review']
    
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
            
            # Show with emoji and color
            if result == "Positive":
                st.success(f"😊 **Sentiment:** {result}")
                st.balloons()
            else:
                st.error(f"😞 **Sentiment:** {result}")
            
            # Show accuracy
            if 'accuracy' in st.session_state:
                st.info(f"Model accuracy: {st.session_state['accuracy']:.2%}")
    
    # Show info if no model trained yet
    if 'accuracy' not in st.session_state:
        st.info("👈 Upload a CSV file and train the model first")

# Add download example CSV option in sidebar
with st.sidebar:
    st.markdown("---")
    st.subheader("📥 Need a dataset?")
    
    # Create example CSV
    example_data = {
        'review': [
            "This movie was amazing! Great acting and story.",
            "Terrible film, waste of time and money.",
            "One of the best movies I've seen this year!",
            "Not worth watching. Poor direction.",
            "Excellent cinematography and performances."
        ],
        'sentiment': ['positive', 'negative', 'positive', 'negative', 'positive']
    }
    
    example_df = pd.DataFrame(example_data)
    
    # Convert to CSV
    csv = example_df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="Download Example CSV",
        data=csv,
        file_name="imdb_example.csv",
        mime="text/csv",
        help="Download a sample CSV file to test the app"
    )
    
    st.markdown("""
    **CSV Format:**
    ```
    review,sentiment
    "Great movie!",positive
    "Bad acting",negative
    ```
    """)
