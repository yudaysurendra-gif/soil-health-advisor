"""
Soil Health Advisor
-------------------
A simple base project: enter soil test values, get a soil health class
(Poor / Moderate / Good), a health score, and actionable recommendations.

Run:  streamlit run app.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------------------------
# 1. Soil parameters: (label, min, max, default, optimal_low, optimal_high)
# ---------------------------------------------------------------------------
PARAMS = {
    "ph":        ("Soil pH",                  4.0,   9.5,   6.5,   6.0,   7.5),
    "nitrogen":  ("Nitrogen (kg/ha)",        50.0, 450.0, 250.0, 280.0, 400.0),
    "phosphorus": ("Phosphorus (kg/ha)",      5.0,  80.0,  20.0,  11.0,  25.0),
    "potassium": ("Potassium (kg/ha)",       50.0, 500.0, 200.0, 120.0, 280.0),
    "carbon":    ("Organic Carbon (%)",       0.1,   2.0,   0.6,   0.5,   0.75),
    "moisture":  ("Soil Moisture (%)",        5.0,  40.0,  20.0,  15.0,  30.0),
    "ec":        ("Electrical Conductivity (dS/m)", 0.0, 4.0, 0.8, 0.0, 1.0),
}
FEATURES = list(PARAMS.keys())
CLASSES = ["Poor", "Moderate", "Good"]


# ---------------------------------------------------------------------------
# 2. Rule-based scoring (used to label training data and to explain results)
# ---------------------------------------------------------------------------
def param_score(key, value):
    """Return 1.0 inside the optimal range, falling to 0 as value moves away."""
    _, lo_lim, hi_lim, _, opt_lo, opt_hi = PARAMS[key]
    if opt_lo <= value <= opt_hi:
        return 1.0
    if value < opt_lo:
        span = max(opt_lo - lo_lim, 1e-9)
        return max(0.0, 1 - (opt_lo - value) / span)
    span = max(hi_lim - opt_hi, 1e-9)
    return max(0.0, 1 - (value - opt_hi) / span)


def health_score(row):
    """Average of individual scores, scaled 0-100."""
    return 100 * np.mean([param_score(k, row[k]) for k in FEATURES])


def score_to_class(score):
    if score >= 80:
        return "Good"
    if score >= 55:
        return "Moderate"
    return "Poor"


# ---------------------------------------------------------------------------
# 3. Synthetic dataset + ML model
#    (Replace generate_data() with a real soil-test CSV when you have one.)
# ---------------------------------------------------------------------------
def generate_data(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    data = {
        k: rng.uniform(PARAMS[k][1], PARAMS[k][2], n) for k in FEATURES
    }
    df = pd.DataFrame(data)
    df["score"] = df.apply(health_score, axis=1)
    df["label"] = df["score"].apply(score_to_class)
    return df


def train_model():
    df = generate_data()
    X, y = df[FEATURES], df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc


# ---------------------------------------------------------------------------
# 4. Recommendation engine
# ---------------------------------------------------------------------------
def recommendations(v):
    tips = []

    if v["ph"] < 6.0:
        tips.append("🟠 Soil is acidic: apply agricultural lime or dolomite to raise pH.")
    elif v["ph"] > 7.5:
        tips.append("🟠 Soil is alkaline: add gypsum, elemental sulphur or organic compost.")

    if v["nitrogen"] < 280:
        tips.append("🟠 Low nitrogen: apply urea/FYM or grow legumes (green gram, cowpea) in rotation.")
    elif v["nitrogen"] > 400:
        tips.append("🟠 Excess nitrogen: reduce urea dose to avoid leaching and pollution.")

    if v["phosphorus"] < 11:
        tips.append("🟠 Low phosphorus: apply DAP or single super phosphate near the root zone.")
    elif v["phosphorus"] > 25:
        tips.append("🟠 High phosphorus: skip phosphatic fertilizer this season.")

    if v["potassium"] < 120:
        tips.append("🟠 Low potassium: apply muriate of potash (MOP) or wood ash.")
    elif v["potassium"] > 280:
        tips.append("🟠 High potassium: no potash needed; watch magnesium balance.")

    if v["carbon"] < 0.5:
        tips.append("🟠 Low organic carbon: add compost, vermicompost, green manure; use mulching.")

    if v["moisture"] < 15:
        tips.append("🟠 Dry soil: irrigate soon and use mulch or drip irrigation to retain water.")
    elif v["moisture"] > 30:
        tips.append("🟠 Waterlogged risk: improve drainage and reduce irrigation frequency.")

    if v["ec"] > 1.0:
        tips.append("🟠 High salinity: leach with good-quality water, add gypsum, grow salt-tolerant crops.")

    if not tips:
        tips.append("🟢 All parameters are in the optimal range. Maintain crop rotation and organic matter.")
    return tips


def suitable_crops(v):
    """Very simple crop hints based on pH and moisture."""
    crops = []
    if 6.0 <= v["ph"] <= 7.5:
        crops += ["Wheat", "Maize", "Tomato", "Cotton"]
    if v["ph"] < 6.0:
        crops += ["Tea", "Potato", "Groundnut"]
    if v["ph"] > 7.5:
        crops += ["Barley", "Sugar beet", "Cotton"]
    if v["moisture"] > 28:
        crops += ["Rice", "Sugarcane"]
    return sorted(set(crops))


# ---------------------------------------------------------------------------
# 5. Streamlit user interface
# ---------------------------------------------------------------------------
def main():
    import streamlit as st

    st.set_page_config(page_title="Soil Health Advisor", page_icon="🌱")
    st.title("🌱 Soil Health Advisor")
    st.caption("Enter your soil test values to get a health rating and advice.")

    @st.cache_resource
    def load_model():
        return train_model()

    model, acc = load_model()

    st.sidebar.header("Soil Test Values")
    values = {}
    for key, (label, lo, hi, default, _, _) in PARAMS.items():
        step = 0.1 if hi <= 10 else 1.0
        values[key] = st.sidebar.slider(label, float(lo), float(hi), float(default), step)

    if st.sidebar.button("Analyze Soil", type="primary"):
        X = pd.DataFrame([values])[FEATURES]
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        score = health_score(values)

        icon = {"Good": "🟢", "Moderate": "🟡", "Poor": "🔴"}[prediction]
        st.subheader(f"{icon} Soil Health: {prediction}")
        c1, c2 = st.columns(2)
        c1.metric("Health Score", f"{score:.0f} / 100")
        c2.metric("Model Confidence", f"{max(proba) * 100:.0f}%")

        st.progress(int(score))

        st.markdown("### Parameter Scores")
        scores_df = pd.DataFrame({
            "Parameter": [PARAMS[k][0] for k in FEATURES],
            "Score (%)": [round(param_score(k, values[k]) * 100) for k in FEATURES],
        }).set_index("Parameter")
        st.bar_chart(scores_df)

        st.markdown("### Recommendations")
        for tip in recommendations(values):
            st.write(tip)

        crops = suitable_crops(values)
        if crops:
            st.markdown("### Suitable Crops")
            st.write(", ".join(crops))
    else:
        st.info("Adjust the values in the sidebar and click **Analyze Soil**.")

    st.sidebar.caption(f"Model accuracy on test data: {acc * 100:.1f}%")


if __name__ == "__main__":
    main()
