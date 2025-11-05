import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# 1. Train models inside the app and cache them
@st.cache_resource
def train_models():
    # Load data
    df = pd.read_csv("harry_potter_1000_students.csv")

    # Features and target
    feature_cols = [
        "Blood Status",
        "Bravery",
        "Intelligence",
        "Loyalty",
        "Ambition",
        "Dark Arts Knowledge",
        "Quidditch Skills",
        "Dueling Skills",
        "Creativity"
    ]
    target_col = "House"

    X = df[feature_cols]
    y = df[target_col]

    # Encode target labels (Gryffindor, etc. -> 0,1,2,3)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    # Preprocessing
    categorical_features = ["Blood Status"]
    numeric_features = [
        "Bravery",
        "Intelligence",
        "Loyalty",
        "Ambition",
        "Dark Arts Knowledge",
        "Quidditch Skills",
        "Dueling Skills",
        "Creativity"
    ]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    # Define three models
    lr_clf = Pipeline(steps=[
        ("preprocess", preprocess),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    rf_clf = Pipeline(steps=[
        ("preprocess", preprocess),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42,
        )),
    ])

    xgb_clf = Pipeline(steps=[
        ("preprocess", preprocess),
        ("clf", XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            random_state=42,
            eval_metric="mlogloss",
        )),
    ])

    # Fit the models
    lr_clf.fit(X_train, y_train)
    rf_clf.fit(X_train, y_train)
    xgb_clf.fit(X_train, y_train)

    return lr_clf, rf_clf, xgb_clf, le


# Train models once (cached by Streamlit)
lr_model, rf_model, xgb_model, le = train_models()

# 2. Streamlit UI
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

col1, col2 = st.columns(2)

with col1:
    blood_status = st.selectbox(
        "Blood Status",
        ["Half-blood", "Muggle-born", "Pure-blood"],
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

input_df = pd.DataFrame([{
    "Blood Status": blood_status,
    "Bravery": bravery,
    "Intelligence": intelligence,
    "Loyalty": loyalty,
    "Ambition": ambition,
    "Dark Arts Knowledge": dark_arts,
    "Quidditch Skills": quidditch,
    "Dueling Skills": dueling,
    "Creativity": creativity,
}])

st.markdown("**Input Summary:**")
st.dataframe(input_df)

if st.button("🔮 Predict House"):
    # Predictions are numeric (0,1,2,3)
    lr_pred_encoded = lr_model.predict(input_df)[0]
    rf_pred_encoded = rf_model.predict(input_df)[0]
    xgb_pred_encoded = xgb_model.predict(input_df)[0]

    # Convert back to house names
    lr_pred = le.inverse_transform([lr_pred_encoded])[0]
    rf_pred = le.inverse_transform([rf_pred_encoded])[0]
    xgb_pred = le.inverse_transform([xgb_pred_encoded])[0]

    st.subheader("Prediction Results")
    st.write(f"**Logistic Regression Prediction:** {lr_pred}")
    st.write(f"**Random Forest Prediction:** {rf_pred}")
    st.write(f"**XGBoost Prediction:** {xgb_pred} 🎯")

    preds = [lr_pred, rf_pred, xgb_pred]
    final_pred = max(set(preds), key=preds.count)

    st.markdown("---")
    st.markdown(f"### Final Voting Result: **{final_pred}** 🧙‍♂️")

    with st.expander("Show XGBoost Class Probabilities"):
        proba = xgb_model.predict_proba(input_df)[0]
        class_indices = xgb_model.named_steps["clf"].classes_
        class_labels = le.inverse_transform(class_indices)

        proba_df = pd.DataFrame({
            "House": class_labels,
            "Probability": proba,
        }).sort_values("Probability", ascending=False)

        st.dataframe(proba_df.reset_index(drop=True))

