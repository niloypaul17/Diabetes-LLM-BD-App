import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

import rag_chatbot
import nearby_care
import export_report

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "diabetes_prediction_model.pkl"
THRESHOLD_PATH = MODEL_DIR / "best_threshold.pkl"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.pkl"
FEATURE_METADATA_PATH = MODEL_DIR / "feature_metadata.pkl"

st.set_page_config(page_title="DiaLLM-BD", page_icon="🩺", layout="wide")

APP_CSS = """
<style>
.stApp { background-color: #f7f9fc; }
h1, h2, h3 { color: #1a2b4a; }
.metric-card { background:#ffffff; border:1px solid #e3e8f0; border-radius:12px; padding:16px 20px; box-shadow:0 1px 3px rgba(20,40,80,0.06); }
.metric-card h3 { color:#2563eb; margin:4px 0 0; }
.risk-badge-high { background:#fde8e8; color:#c1121f; padding:6px 16px; border-radius:20px; font-weight:700; }
.risk-badge-low { background:#e3f7ec; color:#15803d; padding:6px 16px; border-radius:20px; font-weight:700; }
.factor-row { padding:5px 0; color:#233047; }
.source-pill { display:inline-block; background:#eef4ff; color:#2563eb; border:1px solid #cfe0ff; border-radius:14px; padding:3px 12px; margin:3px; font-size:0.82em; text-decoration:none; }
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_prediction_model():
    missing_files = []
    for path, label in [
        (MODEL_PATH, "models/diabetes_prediction_model.pkl"),
        (THRESHOLD_PATH, "models/best_threshold.pkl"),
        (FEATURE_COLUMNS_PATH, "models/feature_columns.pkl"),
        (FEATURE_METADATA_PATH, "models/feature_metadata.pkl"),
    ]:
        if not path.exists():
            missing_files.append(label)

    if missing_files:
        st.error("Model files are missing or stale. Train the model first.")
        st.write("Open a terminal in this folder and run:")
        st.code("python train_model.py", language="bash")
        st.write("Missing files:")
        for file in missing_files:
            st.write(f"- {file}")
        st.stop()

    model = joblib.load(MODEL_PATH)
    threshold = joblib.load(THRESHOLD_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    feature_metadata = joblib.load(FEATURE_METADATA_PATH)
    return model, threshold, feature_columns, feature_metadata


def predict_probability(model, input_df):
    return model.predict_proba(input_df)[:, 1][0]


def get_shap_contributions(pipeline, input_df, categorical_features, top_n=5):
    """Per-patient SHAP contributions, collapsed back to original columns."""
    try:
        import shap
    except ImportError:
        return None, "SHAP is not installed. Run: pip install shap"

    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["classifier"]
        transformed = preprocessor.transform(input_df)
        feature_names = list(preprocessor.get_feature_names_out())

        explainer = shap.TreeExplainer(classifier)
        raw = explainer.shap_values(transformed)
        if isinstance(raw, list):
            raw = raw[1] if len(raw) > 1 else raw[0]
        row = raw[0]

        contributions, display_values = {}, {}
        for fname, val in zip(feature_names, row):
            base = fname.split("__")[-1]
            matched_original = None
            for cat_col in categorical_features:
                if base.startswith(cat_col + "_"):
                    matched_original = cat_col
                    break
            key = matched_original or base
            contributions[key] = contributions.get(key, 0.0) + float(val)
            if key in input_df.columns:
                display_values[key] = input_df.iloc[0][key]

        factors = [
            {"feature": k, "value": display_values.get(k, ""), "contribution": v}
            for k, v in contributions.items()
        ]
        factors.sort(key=lambda f: abs(f["contribution"]), reverse=True)
        return factors[:top_n], None
    except Exception as e:
        return None, f"Explanation unavailable for this model: {e}"


model, threshold, feature_columns, feature_metadata = load_prediction_model()

st.title("🩺 DiaLLM-BD")
st.subheader("Explainable Type-2 Diabetes Risk Screening Support (Research Prototype)")
st.warning("This is a research prototype only. It does not provide final medical diagnosis or treatment.")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Patient Input", "Prediction Explanation", "Diabetes Chatbot", "Nearby Doctors & Hospitals"]
)

# ---------------------------------------------------------------- Tab 1 ----
with tab1:
    st.header("Enter Patient Information")
    patient_input = {}
    numeric_features = feature_metadata.get("numeric_features", [])
    categorical_features = feature_metadata.get("categorical_features", [])
    binary_yesno_features = [
        f for f in ["family_diabetes", "hypertensive", "family_hypertension", "cardiovascular_disease", "stroke"]
        if f in feature_columns
    ]

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        for index, feature in enumerate(feature_columns):
            active_col = col1 if index % 2 == 0 else col2
            with active_col:
                if feature in binary_yesno_features:
                    default_value = feature_metadata["numeric_defaults"].get(feature, 0.0)
                    default_index = 1 if default_value >= 0.5 else 0
                    choice = st.selectbox(label=feature, options=["No", "Yes"], index=default_index)
                    patient_input[feature] = 1 if choice == "Yes" else 0
                elif feature in numeric_features:
                    default_value = feature_metadata["numeric_defaults"].get(feature, 0.0)
                    patient_input[feature] = st.number_input(label=feature, value=float(default_value), step=0.1)
                elif feature in categorical_features:
                    options = feature_metadata["categorical_options"].get(feature, ["Unknown"])
                    default_value = feature_metadata["categorical_defaults"].get(feature, options[0])
                    default_index = options.index(default_value) if default_value in options else 0
                    patient_input[feature] = st.selectbox(label=feature, options=options, index=default_index)
                else:
                    patient_input[feature] = st.text_input(label=feature, value="")
        submitted = st.form_submit_button("Predict Diabetes Risk")

    if submitted:
        input_df = pd.DataFrame([patient_input])
        probability = predict_probability(model, input_df)
        prediction = int(probability >= threshold)

        st.session_state["patient_input"] = patient_input
        st.session_state["probability"] = probability
        st.session_state["prediction"] = prediction

        if prediction == 1:
            st.error(f"Higher Diabetes Risk Detected | Probability: {probability:.2f}")
        else:
            st.success(f"Lower Diabetes Risk Detected | Probability: {probability:.2f}")

# ---------------------------------------------------------------- Tab 2 ----
with tab2:
    st.header("Prediction Explanation")

    if "prediction" not in st.session_state:
        st.info("Please enter patient information and click Predict Diabetes Risk first.")
    else:
        prediction = st.session_state["prediction"]
        probability = st.session_state["probability"]
        patient_input = st.session_state["patient_input"]
        prediction_text = "Higher diabetes risk" if prediction == 1 else "Lower diabetes risk"

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.caption("Estimated probability")
            st.markdown(f"### {probability:.3f}")
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.caption("Decision threshold")
            st.markdown(f"### {threshold:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.caption("Classification")
            badge_class = "risk-badge-high" if prediction == 1 else "risk-badge-low"
            badge_text = "Elevated risk" if prediction == 1 else "Lower risk"
            st.markdown(f'<span class="{badge_class}">{badge_text}</span>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.progress(float(min(max(probability, 0.0), 1.0)))
        st.caption(
            f"The bar shows the estimated probability ({probability:.3f}) on a 0-1 scale. "
            f"The screening threshold is {threshold:.2f}, chosen to favor recall (catching more "
            "possible cases) over precision. The cost of that choice is more false alarms."
        )

        if prediction == 1:
            st.error("The model places this profile above the screening threshold. This is a statistical signal to seek a clinical test, not a diagnosis of diabetes.")
        else:
            st.success("The model places this profile below the screening threshold. This does not rule out diabetes; consult a doctor if symptoms are present.")

        st.subheader("Which values influenced this estimate")
        categorical_features = feature_metadata.get("categorical_features", [])
        input_df = pd.DataFrame([patient_input])
        top_factors, shap_error = get_shap_contributions(model, input_df, categorical_features, top_n=5)

        if shap_error:
            st.info(shap_error)
        else:
            for f in top_factors:
                direction = "increased" if f["contribution"] >= 0 else "decreased"
                st.markdown(
                    f'<div class="factor-row">• <b>{f["feature"]}</b> (your value: {f["value"]}) '
                    f'{direction} the estimated risk (contribution {f["contribution"]:+.3f})</div>',
                    unsafe_allow_html=True,
                )

            fig = export_report.make_waterfall_figure(top_factors, probability, threshold)
            st.pyplot(fig)
            st.caption(
                "Red bars increased this estimate; blue bars decreased it. These are SHAP values: "
                "they show how each recorded value moved this particular estimate relative to the "
                "average patient in the training data. They do not show that any single value "
                "caused diabetes, and they are not medical advice."
            )

            st.divider()
            st.subheader("Download this result")
            fmt = st.selectbox("Format", ["PDF", "DOCX", "PNG", "JPEG"])
            if st.button("Generate report"):
                try:
                    if fmt == "PDF":
                        data = export_report.export_pdf(patient_input, probability, threshold, prediction_text, top_factors)
                        st.download_button("Download PDF", data, file_name="diallm_bd_report.pdf", mime="application/pdf")
                    elif fmt == "DOCX":
                        data = export_report.export_docx(patient_input, probability, threshold, prediction_text, top_factors)
                        st.download_button("Download DOCX", data, file_name="diallm_bd_report.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
                    else:
                        data = export_report.export_png_or_jpeg(patient_input, probability, threshold, prediction_text, top_factors, fmt=fmt.lower())
                        st.download_button(f"Download {fmt}", data, file_name=f"diallm_bd_report.{fmt.lower()}", mime=f"image/{fmt.lower()}")
                except ImportError as e:
                    st.error(f"Missing export dependency: {e}. See README.md for `pip install` steps.")

        st.write("Clinical note: A healthcare professional should confirm diabetes using proper medical tests such as fasting blood glucose, HbA1c, or oral glucose tolerance test.")

# ---------------------------------------------------------------- Tab 3 ----
with tab3:
    st.header("Diabetes Education Chatbot")
    st.caption(
        "Answers are grounded in a local knowledge base (WHO / IDF / ADA fact sheets) and always "
        "cite their source. If a local Ollama model is running, it drafts the reply from that same "
        "retrieved context; otherwise a template assembles the answer directly from the sources."
    )

    question = st.session_state.get("chat_question", "")
    with st.form("chatbot_form", clear_on_submit=False):
        question = st.text_input("Ask a diabetes-related question:", placeholder="Example: What food should a diabetic patient avoid?")
        asked = st.form_submit_button("Ask Chatbot")

    if asked:
        if "prediction" in st.session_state:
            probability = st.session_state["probability"]
            prediction = st.session_state["prediction"]
            prediction_context = f"{'Higher' if prediction == 1 else 'Lower'} diabetes risk, probability {probability:.2f}"
        else:
            prediction_context = "No prediction has been made yet"

        result = rag_chatbot.answer_question(question, prediction_context)
        st.success(result["answer"])
        st.caption(f"Engine: {result['engine']}")
        if result["sources"]:
            st.markdown("**Sources:** " + " ".join(
                f'<a class="source-pill" href="{s["source_url"]}" target="_blank">{s["source_name"]}</a>'
                for s in result["sources"]
            ), unsafe_allow_html=True)

# ---------------------------------------------------------------- Tab 4 ----
with tab4:
    st.header("Nearby Doctors & Hospitals")
    st.caption("Live lookup via OpenStreetMap (Nominatim + Overpass). No account or API key required.")

    use_geo = False
    try:
        from streamlit_js_eval import get_geolocation
        use_geo = st.checkbox("Use my current location (browser geolocation)", value=False)
    except ImportError:
        st.info(
            "Typing your area below works fully on its own — nothing else required. "
            "If you'd rather have a one-click 'use my current location' button instead of typing, "
            "run `pip install streamlit-js-eval` in your venv and restart the app; that adds the checkbox above."
        )

    lat = lon = None
    if use_geo:
        loc = get_geolocation()
        if loc and "coords" in loc:
            lat, lon = loc["coords"]["latitude"], loc["coords"]["longitude"]
            st.success(f"Detected location: {lat:.4f}, {lon:.4f}")

    if lat is None:
        place = st.text_input("Your area / city (e.g. Dhanmondi, Dhaka)", value="Dhanmondi, Dhaka")
        radius_km = st.slider("Search radius (km)", 1, 15, 5)
        if st.button("Find nearby care"):
            geocoded = nearby_care.geocode_place(place)
            if not geocoded:
                st.error("Could not find that location. Try a more specific area name.")
            else:
                lat, lon = geocoded["lat"], geocoded["lon"]
                st.caption(f"Resolved to: {geocoded['display_name']}")
                results = nearby_care.find_nearby_care(lat, lon, radius_km)
                if not results:
                    st.warning("No hospitals/clinics/pharmacies found in OpenStreetMap data for this radius.")
                else:
                    st.map(pd.DataFrame([{"lat": r["lat"], "lon": r["lon"]} for r in results] + [{"lat": lat, "lon": lon}]))
                    st.dataframe(pd.DataFrame(results)[["name", "type", "distance_km", "phone"]].round({"distance_km": 2}))
    else:
        radius_km = st.slider("Search radius (km)", 1, 15, 5, key="radius_geo")
        results = nearby_care.find_nearby_care(lat, lon, radius_km)
        if not results:
            st.warning("No hospitals/clinics/pharmacies found in OpenStreetMap data for this radius.")
        else:
            st.map(pd.DataFrame([{"lat": r["lat"], "lon": r["lon"]} for r in results] + [{"lat": lat, "lon": lon}]))
            st.dataframe(pd.DataFrame(results)[["name", "type", "distance_km", "phone"]].round({"distance_km": 2}))

    st.caption("Data from OpenStreetMap contributors. Coverage varies by area; always call ahead to confirm services and hours.")

st.markdown("---")
st.caption("DiaLLM-BD: Machine Learning + retrieval-grounded patient education prototype.")
