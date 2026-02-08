import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("📧 Spam Email Classifier")

# Create two columns
col1, col2 = st.columns(2)

# Dataset
emails = [
    "Congratulations! You've won a free iPhone",
    "Claim your lottery prize now",
    "Exclusive deal just for you",
    "Act fast! Limited-time offer",
    "Click here to secure your reward",
    "Win cash prizes instantly by signing up",
    "Limited-time discount on luxury watches",
    "Get rich quick with this secret method",
    "Hello, how are you today",
    "Please find the attached report",
    "Thank you for your support",
    "The project deadline is next week",
    "Can we reschedule the meeting to tomorrow",
    "Your invoice for last month is attached",
    "Looking forward to our call later today",
    "Don't forget the team lunch tomorrow",
    "Meeting agenda has been updated",
    "Here are the notes from yesterday's discussion",
    "Please confirm your attendance for the workshop",
    "Let's finalize the budget proposal by Friday"
]

labels = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Left column: Dataset and Model Training
with col1:
    st.subheader("📊 Dataset")
    
    # Show dataset in a table
    data_display = []
    for i, (email, label) in enumerate(zip(emails, labels)):
        data_display.append({
            "Email": email,
            "Label": "Spam" if label == 1 else "Not Spam"
        })
    
    st.dataframe(data_display)
    
    if st.button("Train Model", key="train"):
        with st.spinner("Training model..."):
            # Vectorize emails
            vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words='english',
                ngram_range=(1,2),
                max_df=0.9,
                min_df=1
            )
            X = vectorizer.fit_transform(emails)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, labels, test_size=0.25, random_state=42, stratify=labels
            )
            
            # Train model
            svm_model = LinearSVC(C=1.0)
            svm_model.fit(X_train, y_train)
            
            # Predict and calculate accuracy
            y_pred = svm_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Store in session state
            st.session_state['vectorizer'] = vectorizer
            st.session_state['model'] = svm_model
            st.session_state['accuracy'] = accuracy
            
            st.success(f"✅ Model trained successfully!")
            st.metric("Model Accuracy", f"{accuracy:.2%}")

# Right column: Prediction
with col2:
    st.subheader("🔮 Spam Detection")
    
    # Text area for email input
    new_email = st.text_area(
        "Enter email text to check:",
        height=150,
        placeholder="Paste or type email content here..."
    )
    
    if st.button("Check if Spam", key="predict"):
        if 'model' not in st.session_state:
            st.warning("⚠️ Please train the model first!")
        elif not new_email.strip():
            st.warning("⚠️ Please enter an email text!")
        else:
            # Get model and vectorizer from session state
            vectorizer = st.session_state['vectorizer']
            model = st.session_state['model']
            
            # Vectorize and predict
            new_email_vectorized = vectorizer.transform([new_email])
            prediction = model.predict(new_email_vectorized)
            
            # Display result
            if prediction[0] == 1:
                st.error("🚫 **Result: SPAM Email**")
            else:
                st.success("✅ **Result: NOT SPAM (Ham)**")
            
            # Show accuracy
            if 'accuracy' in st.session_state:
                st.info(f"Model accuracy: {st.session_state['accuracy']:.2%}")