import streamlit as st
import pandas as pd
import numpy as np
import math

st.title("🌳 Decision Tree Classifier")

# Dataset
data = pd.DataFrame({
    'Outlook': ['Sunny','Sunny','Overcast','Rain','Rain','Rain',
                'Overcast','Sunny','Sunny','Rain','Sunny','Overcast',
                'Overcast','Rain'],
    'Humidity': ['High','High','High','High','Normal','Normal',
                 'Normal','High','Normal','High','Normal','High',
                 'Normal','High'],
    'PlayTennis': ['No','No','Yes','Yes','Yes','No',
                    'Yes','No','Yes','Yes','Yes','Yes',
                    'Yes','No']
})

# Create two columns
col1, col2 = st.columns(2)

# Left column: Dataset and Tree
with col1:
    st.subheader("📊 Dataset")
    st.dataframe(data)
    
    if st.button("Generate Decision Tree", key="generate"):
        # Functions
        def entropy(col):
            values, counts = np.unique(col, return_counts=True)
            ent = 0
            for count in counts:
                p = count / len(col)
                ent -= p * math.log2(p)
            return ent

        def information_gain(df, attribute, target):
            total_entropy = entropy(df[target])
            values, counts = np.unique(df[attribute], return_counts=True)

            weighted_entropy = 0
            for i in range(len(values)):
                subset = df[df[attribute] == values[i]]
                weighted_entropy += (counts[i]/len(df)) * entropy(subset[target])

            return total_entropy - weighted_entropy

        def id3(df, target, attributes):
            if len(np.unique(df[target])) == 1:
                return df[target].iloc[0]
            if len(attributes) == 0:
                return df[target].mode()[0]
            
            gains = [information_gain(df, attr, target) for attr in attributes]
            best_attr = attributes[np.argmax(gains)]
            tree = {best_attr: {}}
            
            for value in np.unique(df[best_attr]):
                subset = df[df[best_attr] == value]
                remaining_attrs = [attr for attr in attributes if attr != best_attr]
                tree[best_attr][value] = id3(subset, target, remaining_attrs)
            
            return tree

        # Generate tree
        attributes = list(data.columns)
        attributes.remove('PlayTennis')
        decision_tree = id3(data, 'PlayTennis', attributes)
        
        st.subheader("🌲 Decision Tree")
        st.json(decision_tree)

# Right column: Prediction
with col2:
    st.subheader("🔮 Prediction")
    
    # Input fields
    outlook = st.selectbox("Outlook", ["Sunny", "Overcast", "Rain"], key="outlook")
    humidity = st.selectbox("Humidity", ["High", "Normal"], key="humidity")
    
    if st.button("Predict", key="predict"):
        # Functions (copied for this scope)
        def entropy(col):
            values, counts = np.unique(col, return_counts=True)
            ent = 0
            for count in counts:
                p = count / len(col)
                ent -= p * math.log2(p)
            return ent

        def information_gain(df, attribute, target):
            total_entropy = entropy(df[target])
            values, counts = np.unique(df[attribute], return_counts=True)

            weighted_entropy = 0
            for i in range(len(values)):
                subset = df[df[attribute] == values[i]]
                weighted_entropy += (counts[i]/len(df)) * entropy(subset[target])

            return total_entropy - weighted_entropy

        def id3(df, target, attributes):
            if len(np.unique(df[target])) == 1:
                return df[target].iloc[0]
            if len(attributes) == 0:
                return df[target].mode()[0]
            
            gains = [information_gain(df, attr, target) for attr in attributes]
            best_attr = attributes[np.argmax(gains)]
            tree = {best_attr: {}}
            
            for value in np.unique(df[best_attr]):
                subset = df[df[best_attr] == value]
                remaining_attrs = [attr for attr in attributes if attr != best_attr]
                tree[best_attr][value] = id3(subset, target, remaining_attrs)
            
            return tree

        def predict(tree, sample):
            if not isinstance(tree, dict):
                return tree
            attr = next(iter(tree))
            value = sample[attr]
            if value in tree[attr]:
                return predict(tree[attr][value], sample)
            else:
                return "Unknown"

        # Generate tree and predict
        attributes = list(data.columns)
        attributes.remove('PlayTennis')
        decision_tree = id3(data, 'PlayTennis', attributes)
        
        sample = {'Outlook': outlook, 'Humidity': humidity}
        result = predict(decision_tree, sample)
        
        # Display result
        st.markdown("### Result")
        if result == "Yes":
            st.success(f"✅ **Play Tennis:** {result}")
        elif result == "No":
            st.error(f"❌ **Play Tennis:** {result}")
        else:
            st.warning(f"❓ **Play Tennis:** {result}")
        
        # Show sample
        st.write("**Sample:**")
        st.write(f"- Outlook: {outlook}")
        st.write(f"- Humidity: {humidity}")