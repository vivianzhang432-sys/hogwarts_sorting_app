import streamlit as st
import pandas as pd
import joblib

# 1. Load pre-trained models and label encoder
@st.cache_resource
def load_models():
    lr_model = joblib.load("lr_pipeline.pkl")
    rf_model = joblib.load("rf_pipeline.pkl")
    xgb_model = joblib.load("xgb_pipeline.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return lr_model, rf_model, xgb_model, label_encoder

lr_model, rf_model, xgb_model, le = load_models()

# 2. Page title and introduction
st.title("🏰 Hogwarts Sorting Prediction - Interactive Demo")
st.write(
    """
    This interactive demo uses three machine learning models:
    **Logistic Regression**, **Random Forest**, and **XGBoost**.  
    You can input a student's traits below, and the models will predict
    which Hogwarts House best fits them.
    """
)

st.markdown("### Please enter the student's characteristics:")

# 3. Input fields
col1, col2 = st.columns(2)

with col1:
    blood_status = st.selectbox(
        "Blood Status",
        ["Half-blood", "Muggle-born", "Pure-blood"]
    )
    bravery = st.slider("Bravery", 0, 10, 5)
    intelligence = st.slider("Intelligence", 0, 10, 5)
    loyalty = st.slider("Loyalty", 0, 10, 5)
    ambition = st.slider("Ambition", 0, 10, 5)

with col2:
    dark_arts = st.slider("Dark Arts Knowledge", 0, 10, 5)
    quidditch = st.slider("Quidditch Skills", 0, 10, 5)
    dueling = st.slider("Dueling Skills", 0, 10, 5)
    creativity = st.slider("Creativity", 0, 10, 5)

# 4. Combine input into a DataFrame
input_df = pd.DataFrame([{
    "Blood Status": blood_status,
    "Bravery": bravery,
    "Intelligence": intelligence,
    "Loyalty": loyalty,
    "Ambition": ambition,
    "Dark Arts Knowledge": dark_arts,
    "Quidditch Skills": quidditch,
    "Dueling Skills": dueling,
    "Creativity": creativity
}])

st.markdown("**Input Summary:**")
st.dataframe(input_df)

# 5. Prediction button
if st.button("🔮 Predict House"):
    # Predict using all three models
    lr_pred_encoded = lr_model.predict(input_df)[0]
    rf_pred_encoded = rf_model.predict(input_df)[0]
    xgb_pred_encoded = xgb_model.predict(input_df)[0]

    lr_pred = le.inverse_transform([lr_pred_encoded])[0]
    rf_pred = le.inverse_transform([rf_pred_encoded])[0]
    xgb_pred = le.inverse_transform([xgb_pred_encoded])[0]

    # Display results
    st.subheader("Prediction Results")
    st.write(f"**Logistic Regression Prediction:** {lr_pred}")
    st.write(f"**Random Forest Prediction:** {rf_pred}")
    st.write(f"**XGBoost Prediction:** {xgb_pred} 🎯")

    # Majority voting
    preds = [lr_pred, rf_pred, xgb_pred]
    final_pred = max(set(preds), key=preds.count)

    st.markdown("---")
    st.markdown(f"### Final Voting Result: **{final_pred}** 🧙‍♂️")

    # Show probability details
    with st.expander("Show XGBoost Class Probabilities"):
        proba = xgb_model.predict_proba(input_df)[0]
        class_indices = xgb_model.named_steps["clf"].classes_
        class_labels = le.inverse_transform(class_indices)

        proba_df = pd.DataFrame({
            "House": class_labels,
            "Probability": proba
        }).sort_values("Probability", ascending=False)

        st.dataframe(proba_df.reset_index(drop=True))
