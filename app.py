
import os
import re
import joblib
import streamlit as st
import pandas as pd
import numpy as np

from lime.lime_text import LimeTextExplainer

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="TruthLens AI",
    page_icon="🔎",
    layout="wide"
)

# --------------------------------------------------
# 2. FILE PATHS
# --------------------------------------------------

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "final_model.pkl")

VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "models", "tfidf_vectorizer.pkl")

ENCODER_PATH = os.path.join(os.path.dirname(__file__), "models", "label_encoder.pkl")


# --------------------------------------------------
# 3. LOAD TRAINED MODEL AND PREPROCESSING OBJECTS
# --------------------------------------------------

@st.cache_resource
def load_artifacts():

    model = joblib.load(MODEL_PATH)

    vectorizer = joblib.load(VECTORIZER_PATH)

    label_encoder = joblib.load(ENCODER_PATH)

    return model, vectorizer, label_encoder


try:
    model, tfidf_vectorizer, label_encoder = load_artifacts()

except Exception as error:
    st.error("Could not load the trained model files.")
    st.exception(error)
    st.stop()


# --------------------------------------------------
# 4. TEXT PREPROCESSING
# --------------------------------------------------

def clean_text(text):

    text = str(text)

    text = text.lower()

    text = re.sub(
        r'https?://\S+|www\.\S+',
        '',
        text
    )

    text = re.sub(
        r'\s+',
        ' ',
        text
    )

    text = text.strip()

    return text


# --------------------------------------------------
# 5. PREDICTION FUNCTION
# --------------------------------------------------

def predict_probabilities(text_list):

    cleaned_texts = [
        clean_text(text)
        for text in text_list
    ]

    text_features = tfidf_vectorizer.transform(
        cleaned_texts
    )

    probabilities = model.predict_proba(
        text_features
    )

    return probabilities


# --------------------------------------------------
# 6. CHECK CLASS ORDER
# --------------------------------------------------

class_names = list(label_encoder.classes_)

expected_classes = np.arange(len(class_names))

if not np.array_equal(model.classes_, expected_classes):

    st.error(
        "Model class order does not match the label encoder. "
        "Please check the saved model and label encoder."
    )

    st.stop()


explainer = LimeTextExplainer(
    class_names=class_names,
    random_state=42
)


# --------------------------------------------------
# 7. APPLICATION HEADER
# --------------------------------------------------

st.title("🔎 TruthLens AI")

st.subheader(
    "AI-Based Claim Credibility Classification"
)

st.write(
    """
    TruthLens AI is an experimental machine-learning application
    that classifies short statements into six credibility categories
    using the LIAR dataset.
    """
)

st.info(
    """
    **Research disclaimer:** TruthLens AI does not verify facts or
    determine objective truth. It identifies patterns learned from
    previously fact-checked statements. Always verify important
    claims using reliable sources.
    """
)

st.divider()


# --------------------------------------------------
# 8. USER INPUT
# --------------------------------------------------

st.header("Analyze a statement")

user_statement = st.text_area(
    "Enter a statement:",
    placeholder="Type or paste a statement here...",
    height=150
)

analyze_button = st.button(
    "Analyze Statement",
    type="primary",
    use_container_width=True
)


# --------------------------------------------------
# 9. ANALYSIS
# --------------------------------------------------

if analyze_button:

    if not user_statement.strip():

        st.warning("Please enter a statement first.")

    elif len(user_statement.strip().split()) < 2:

        st.warning(
            "Please enter a longer statement for analysis."
        )

    else:

        with st.spinner(
            "Analyzing statement and generating explanation..."
        ):

            # Get probabilities
            probabilities = predict_probabilities(
                [user_statement]
            )[0]

            # Find predicted class
            predicted_index = int(
                np.argmax(probabilities)
            )

            predicted_category = class_names[
                predicted_index
            ]

            confidence = float(
                probabilities[predicted_index]
            )

            # Generate LIME explanation
            explanation = explainer.explain_instance(
                user_statement,
                classifier_fn=predict_probabilities,
                labels=(predicted_index,),
                num_features=10,
                num_samples=1000
            )

            word_explanations = explanation.as_list(
                label=predicted_index
            )

        # --------------------------------------------------
        # 10. DISPLAY PREDICTION
        # --------------------------------------------------

        st.divider()

        st.header("Analysis Result")

        st.write("**Submitted statement:**")

        st.write(user_statement)

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.metric(
                label="Predicted Category",
                value=predicted_category.replace("-", " ").title()
            )

        with result_col2:

            st.metric(
                label="Model Probability",
                value=f"{confidence * 100:.2f}%"
            )

        st.caption(
            "This probability is the model's estimated score for "
            "the selected category, not the probability that the "
            "statement is objectively true."
        )

        # --------------------------------------------------
        # 11. DISPLAY ALL CLASS PROBABILITIES
        # --------------------------------------------------

        st.divider()

        st.header("Category Probabilities")

        probability_df = pd.DataFrame({
            "Category": [
                category.replace("-", " ").title()
                for category in class_names
            ],

            "Probability (%)": [
                round(float(value) * 100, 2)
                for value in probabilities
            ]
        })

        probability_df = probability_df.sort_values(
            by="Probability (%)",
            ascending=False
        )

        st.dataframe(
            probability_df,
            use_container_width=True,
            hide_index=True
        )

        chart_df = probability_df.set_index(
            "Category"
        )

        st.bar_chart(
            chart_df,
            y="Probability (%)"
        )

        # --------------------------------------------------
        # 12. DISPLAY LIME EXPLANATION
        # --------------------------------------------------

        st.divider()

        st.header("Explainable AI — LIME")

        st.write(
            """
            LIME explains this individual prediction by estimating
            how changes to words in the statement affect the model's
            output near this particular example.
            """
        )

        lime_df = pd.DataFrame(
            word_explanations,
            columns=["Word or Feature", "Influence"]
        )

        lime_df["Influence"] = lime_df[
            "Influence"
        ].round(4)

        st.dataframe(
            lime_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Positive influence supports the selected category; "
            "negative influence pushes the prediction away from it. "
            "These are local model explanations, not evidence that "
            "the statement is true or false."
        )

        # --------------------------------------------------
        # 13. DOWNLOAD RESULTS
        # --------------------------------------------------

        result_csv = probability_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download Probability Results",
            data=result_csv,
            file_name="truthlens_probabilities.csv",
            mime="text/csv"
        )


# --------------------------------------------------
# 14. SIDEBAR
# --------------------------------------------------

st.sidebar.title("About TruthLens AI")

st.sidebar.write(
    """
    **Project:** TruthLens AI

    **Task:** Six-class claim credibility classification

    **Dataset:** LIAR

    **Machine Learning:** Logistic Regression

    **Text Features:** TF-IDF

    **Explainability:** LIME

    **Framework:** Streamlit
    """
)

st.sidebar.divider()

st.sidebar.caption(
    "Developed as an academic AI/ML project."
)

st.sidebar.caption(
    "Experimental research tool — not a fact-checking authority."
)
