# =============================================================================
# app.py — Flask Backend for Credit Card Approval Prediction
# =============================================================================
# Run: python app.py
# Visit: http://127.0.0.1:5000/
# =============================================================================

import os
import pickle
import numpy as np
from flask import Flask, render_template, request, redirect, url_for

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, 'model.pkl')
ENCODERS_PATH = os.path.join(BASE_DIR, 'encoders.pkl')

# Load trained model and encoders at startup
def _load_artifacts():
    """Load model.pkl and encoders.pkl. Returns (model, encoders) or (None, None)."""
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(ENCODERS_PATH, 'rb') as f:
            encoders = pickle.load(f)
        return model, encoders
    except FileNotFoundError:
        return None, None

model, encoders = _load_artifacts()


# ---------------------------------------------------------------------------
# Route helpers
# ---------------------------------------------------------------------------

def _encode_input(form_data: dict) -> np.ndarray:
    """
    Convert raw form values into the exact numeric feature vector
    used during training.  Applies the same LabelEncoders and
    mapping dicts saved in encoders.pkl.
    """
    gender_le   = encoders['gender_le']
    car_le      = encoders['car_le']
    realty_le   = encoders['realty_le']
    housing_map = encoders['housing_map']
    income_map  = encoders['income_map']
    edu_map     = encoders['education_map']
    family_map  = encoders['family_map']

    # Categorical → encoded int
    gender  = int(gender_le.transform([form_data['CODE_GENDER']])[0])
    car     = int(car_le.transform([form_data['FLAG_OWN_CAR']])[0])
    realty  = int(realty_le.transform([form_data['FLAG_OWN_REALTY']])[0])
    housing = housing_map.get(form_data['NAME_HOUSING_TYPE'], 0)
    income  = income_map.get(form_data['NAME_INCOME_TYPE'], 0)
    edu     = edu_map.get(form_data['NAME_EDUCATION_TYPE'], 0)
    family  = family_map.get(form_data['NAME_FAMILY_STATUS'], 0)

    # Numeric fields
    cnt_children     = int(form_data.get('CNT_CHILDREN', 0))
    amt_income       = float(form_data.get('AMT_INCOME_TOTAL', 0))
    days_birth       = abs(float(form_data.get('DAYS_BIRTH', 0)))
    days_employed    = abs(float(form_data.get('DAYS_EMPLOYED', 0)))
    flag_mobil       = int(form_data.get('FLAG_MOBIL', 1))
    flag_work_phone  = int(form_data.get('FLAG_WORK_PHONE', 0))
    flag_phone       = int(form_data.get('FLAG_PHONE', 0))
    flag_email       = int(form_data.get('FLAG_EMAIL', 0))
    cnt_fam          = float(form_data.get('CNT_FAM_MEMBERS', 1))
    family_dep       = cnt_children / max(cnt_fam, 1)

    # Credit-derived features (entered directly in form)
    open_month   = float(form_data.get('open_month', -12))
    end_month    = float(form_data.get('end_month', 0))
    total_loans  = int(form_data.get('total_loans', 1))
    emi_paid_off = int(form_data.get('emi_paid_off', 0))
    emi_overdue  = int(form_data.get('emi_overdue', 0))
    window       = end_month - open_month

    # Build feature vector in the SAME column order as training
    # (feature_cols saved in encoders.pkl)
    feature_cols = encoders.get('feature_cols', [])

    row = {
        'CODE_GENDER'         : gender,
        'FLAG_OWN_CAR'        : car,
        'FLAG_OWN_REALTY'     : realty,
        'CNT_CHILDREN'        : cnt_children,
        'AMT_INCOME_TOTAL'    : amt_income,
        'NAME_INCOME_TYPE'    : income,
        'NAME_EDUCATION_TYPE' : edu,
        'NAME_FAMILY_STATUS'  : family,
        'NAME_HOUSING_TYPE'   : housing,
        'DAYS_BIRTH'          : days_birth,
        'DAYS_EMPLOYED'       : days_employed,
        'FLAG_MOBIL'          : flag_mobil,
        'FLAG_WORK_PHONE'     : flag_work_phone,
        'FLAG_PHONE'          : flag_phone,
        'FLAG_EMAIL'          : flag_email,
        'CNT_FAM_MEMBERS'     : cnt_fam,
        'family_dependency'   : family_dep,
        'open_month'          : open_month,
        'end_month'           : end_month,
        'total_loans'         : total_loans,
        'emi_paid_off'        : emi_paid_off,
        'emi_overdue'         : emi_overdue,
        'window'              : window,
    }

    if feature_cols:
        vector = [row.get(col, 0) for col in feature_cols]
    else:
        vector = list(row.values())

    return np.array(vector).reshape(1, -1)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def home():
    """Landing page."""
    return render_template('home.html')


@app.route('/form')
def form():
    """Prediction input form."""
    if model is None:
        return render_template(
            'result.html',
            prediction_text='⚠️ Model not trained yet. Please run: python train.py',
            approved=False,
            form_data={},
            error=True
        )
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Receive form data → encode → predict → render result.
    Applies the exact same label encoding used during training.
    """
    if model is None:
        return render_template(
            'result.html',
            prediction_text='⚠️ Model not found. Run python train.py first.',
            approved=False,
            form_data={},
            error=True
        )

    try:
        form_data = request.form.to_dict()
        features  = _encode_input(form_data)

        # Predict
        prediction = model.predict(features)
        probability = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)[0]
            probability = round(float(proba[1]) * 100, 1)  # % chance of risk

        approved = (prediction[0] == 0)  # 0 = not risky → approved

        if approved:
            result_text = '✅ Credit Card Approved'
        else:
            result_text = '❌ Credit Card Rejected'

        return render_template(
            'result.html',
            prediction_text=result_text,
            approved=approved,
            probability=probability,
            form_data=form_data,
            error=False
        )

    except Exception as e:
        return render_template(
            'result.html',
            prediction_text=f'⚠️ Prediction error: {str(e)}',
            approved=False,
            form_data={},
            error=True
        )


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    app.run(debug=True)
