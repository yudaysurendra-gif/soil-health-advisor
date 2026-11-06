# Soil Health Advisor

Enter soil test values (pH, N, P, K, organic carbon, moisture, EC) and get a
health class (Poor / Moderate / Good), a 0-100 score, fertilizer/amendment
advice and crop suggestions.

## Run
    pip install -r requirements.txt
    streamlit run app.py

## How it works
1. Each parameter is scored against its optimal range (rule-based).
2. A Random Forest is trained on synthetic labelled data to classify health.
3. A rule engine turns out-of-range values into practical recommendations.

Replace `generate_data()` in app.py with a real soil-test CSV to improve it.
