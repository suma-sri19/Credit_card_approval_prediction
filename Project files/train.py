# =============================================================================
# train.py — Full ML Pipeline for Credit Card Approval Prediction
# =============================================================================
import sys
# Force UTF-8 output on Windows so emoji/box-drawing chars print correctly
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
# Run: python train.py
# Output: model.pkl, encoders.pkl saved in project root
# =============================================================================

import os
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(BASE_DIR, 'data')
PLOT_DIR       = os.path.join(BASE_DIR, 'static', 'assets', 'eda')
MODEL_PATH     = os.path.join(BASE_DIR, 'model.pkl')
ENCODERS_PATH  = os.path.join(BASE_DIR, 'encoders.pkl')

os.makedirs(PLOT_DIR, exist_ok=True)


# ===========================================================================
# STEP 1 — Data Loading
# ===========================================================================

def load_data():
    """
    Load application_record.csv and credit_record.csv from data/.
    Falls back to a synthetic dataset generator when the real CSVs are absent,
    so the pipeline can be demonstrated end-to-end without the Kaggle files.
    """
    app_path    = os.path.join(DATA_DIR, 'application_record.csv')
    credit_path = os.path.join(DATA_DIR, 'credit_record.csv')

    if os.path.exists(app_path) and os.path.exists(credit_path):
        print("[INFO] Loading real Kaggle datasets …")
        app    = pd.read_csv(app_path)
        credit = pd.read_csv(credit_path)
    else:
        print("[WARN] CSV files not found in data/. Generating synthetic data …")
        app, credit = _generate_synthetic_data()

    print(f"  application_record : {app.shape}")
    print(f"  credit_record      : {credit.shape}")
    return app, credit


def _generate_synthetic_data(n_app: int = 5000, n_credit: int = 15000):
    """
    Produce realistic-looking synthetic data that mirrors the Kaggle schema.
    Used as a fallback when real CSVs are absent.
    """
    rng = np.random.default_rng(42)

    ids = np.arange(1, n_app + 1)

    app = pd.DataFrame({
        'ID'                  : ids,
        'CODE_GENDER'         : rng.choice(['M', 'F'], n_app),
        'FLAG_OWN_CAR'        : rng.choice(['Y', 'N'], n_app),
        'FLAG_OWN_REALTY'     : rng.choice(['Y', 'N'], n_app),
        'CNT_CHILDREN'        : rng.integers(0, 5, n_app),
        'AMT_INCOME_TOTAL'    : rng.uniform(30000, 500000, n_app).round(2),
        'NAME_INCOME_TYPE'    : rng.choice(
            ['Working', 'Commercial associate', 'Pensioner', 'State servant'], n_app),
        'NAME_EDUCATION_TYPE' : rng.choice(
            ['Higher education', 'Secondary / secondary special',
             'Incomplete higher', 'Lower secondary'], n_app),
        'NAME_FAMILY_STATUS'  : rng.choice(
            ['Married', 'Single / not married', 'Civil marriage',
             'Separated', 'Widow'], n_app),
        'NAME_HOUSING_TYPE'   : rng.choice(
            ['House / apartment', 'Rented apartment', 'With parents',
             'Municipal apartment', 'Co-op apartment'], n_app),
        'DAYS_BIRTH'          : -rng.integers(7000, 25000, n_app),
        'DAYS_EMPLOYED'       : -rng.integers(0, 15000, n_app),
        'FLAG_MOBIL'          : rng.integers(0, 2, n_app),
        'FLAG_WORK_PHONE'     : rng.integers(0, 2, n_app),
        'FLAG_PHONE'          : rng.integers(0, 2, n_app),
        'FLAG_EMAIL'          : rng.integers(0, 2, n_app),
        'OCCUPATION_TYPE'     : rng.choice(
            ['Laborers', 'Core staff', 'Accountants', 'Managers', np.nan], n_app,
            p=[0.3, 0.2, 0.1, 0.1, 0.3]),
        'CNT_FAM_MEMBERS'     : rng.integers(1, 7, n_app).astype(float),
    })

    # Credit records — multiple rows per applicant
    credit_ids = rng.choice(ids, n_credit, replace=True)
    statuses   = rng.choice(['C', 'X', '0', '1', '2', '3', '4', '5'],
                             n_credit, p=[0.35, 0.25, 0.15, 0.1, 0.06, 0.04, 0.03, 0.02])
    months     = rng.integers(-60, 0, n_credit)

    credit = pd.DataFrame({
        'ID'             : credit_ids,
        'MONTHS_BALANCE' : months,
        'STATUS'         : statuses,
    })

    return app, credit


# ===========================================================================
# STEP 2 — Exploratory Data Analysis (saves plots to static/assets/eda/)
# ===========================================================================

def run_eda(app: pd.DataFrame):
    """
    Generate and save EDA plots:
      - Countplots for categorical columns
      - Correlation heatmap for numeric columns
    """
    print("\n[STEP 2] Running EDA …")

    cat_cols = ['NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
                'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE']

    for col in cat_cols:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(
            x=col, data=app,
            palette='Set2', ax=ax,
            order=app[col].value_counts().index
        )
        ax.set_title(f'Distribution of {col}', fontsize=14, fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        fname = os.path.join(PLOT_DIR, f'{col.lower()}_dist.png')
        fig.savefig(fname, dpi=100, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {fname}")

    # Correlation heatmap (numeric columns only)
    numeric_df = app.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        numeric_df.corr(), annot=True, fmt='.2f',
        cmap='coolwarm', linewidths=0.5, ax=ax
    )
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    heatmap_path = os.path.join(PLOT_DIR, 'correlation_heatmap.png')
    fig.savefig(heatmap_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {heatmap_path}")

    print("[INFO] Descriptive statistics:")
    print(app.describe().to_string())


# ===========================================================================
# STEP 3 — Data Preprocessing
# ===========================================================================

def is_risky(status: str) -> int:
    """
    Convert raw STATUS code into binary risk label.
    STATUS codes:
      0 = 1-29 days overdue   → low risk
      1 = 30-59 days overdue  → medium risk
      2 = 60-89 days          → HIGH RISK
      3 = 90-119 days         → HIGH RISK
      4 = 120-149 days        → HIGH RISK
      5 = bad debt / write-off → HIGH RISK
      C = paid off that month → good
      X = no loan that month  → neutral
    Returns 1 (risky → reject) or 0 (good → approve).
    """
    return 1 if str(status) in ['2', '3', '4', '5'] else 0


# --- Numeric maps for categorical features --------------------------------
_HOUSING_MAP = {
    'House / apartment'  : 0,
    'Rented apartment'   : 1,
    'With parents'       : 2,
    'Municipal apartment': 3,
    'Co-op apartment'    : 4,
    'Office apartment'   : 5,
}

_INCOME_MAP = {
    'Working'               : 0,
    'Commercial associate'  : 1,
    'Pensioner'             : 2,
    'State servant'         : 3,
    'Student'               : 4,
}

_EDUCATION_MAP = {
    'Higher education'                : 0,
    'Secondary / secondary special'   : 1,
    'Incomplete higher'               : 2,
    'Lower secondary'                 : 3,
    'Academic degree'                 : 4,
}

_FAMILY_MAP = {
    'Married'            : 0,
    'Single / not married': 1,
    'Civil marriage'     : 2,
    'Separated'          : 3,
    'Widow'              : 4,
}


def data_cleaning(app: pd.DataFrame, credit: pd.DataFrame):
    """
    Full preprocessing pipeline:
      1. Remove duplicates from application records
      2. Handle missing values (drop OCCUPATION_TYPE)
      3. Fix negative DAYS_BIRTH / DAYS_EMPLOYED with abs()
      4. Engineer family_dependency feature
      5. Map categorical columns to numeric labels
      6. Aggregate credit records per applicant
      7. Engineer open_month, end_month, window features
      8. Create binary target label using is_risky()
      9. Merge application + credit datasets
    Returns: merged DataFrame ready for modelling
    """
    print("\n[STEP 3] Data Cleaning & Feature Engineering …")

    # ── 3.1 Deduplicate application records ──────────────────────────────
    subset_cols = [
        'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
        'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE',
        'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
        'DAYS_BIRTH', 'DAYS_EMPLOYED', 'FLAG_MOBIL', 'FLAG_WORK_PHONE',
        'FLAG_PHONE', 'FLAG_EMAIL', 'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS',
    ]
    before = len(app)
    app.drop_duplicates(subset=subset_cols, keep='first', inplace=True)
    print(f"  Duplicates removed : {before - len(app)}")

    # ── 3.2 Handle missing values ─────────────────────────────────────────
    print(f"  Missing values before:\n{app.isnull().sum()[app.isnull().sum() > 0]}")
    app.drop(columns=['OCCUPATION_TYPE'], inplace=True, errors='ignore')
    print(f"  Dropped OCCUPATION_TYPE column (high missing rate)")

    # ── 3.3 Fix negative days ─────────────────────────────────────────────
    app['DAYS_BIRTH']     = app['DAYS_BIRTH'].abs()
    app['DAYS_EMPLOYED']  = app['DAYS_EMPLOYED'].abs()

    # ── 3.4 Family dependency feature ────────────────────────────────────
    app['family_dependency'] = app['CNT_CHILDREN'] / app['CNT_FAM_MEMBERS'].replace(0, 1)

    # ── 3.5 Map categorical columns to numeric ────────────────────────────
    app['NAME_HOUSING_TYPE']   = app['NAME_HOUSING_TYPE'].map(_HOUSING_MAP).fillna(0)
    app['NAME_INCOME_TYPE']    = app['NAME_INCOME_TYPE'].map(_INCOME_MAP).fillna(0)
    app['NAME_EDUCATION_TYPE'] = app['NAME_EDUCATION_TYPE'].map(_EDUCATION_MAP).fillna(0)
    app['NAME_FAMILY_STATUS']  = app['NAME_FAMILY_STATUS'].map(_FAMILY_MAP).fillna(0)

    # ── 3.6 Aggregate credit records per applicant ────────────────────────
    # Risk label per applicant (1 = risky, 0 = good)
    credit['risk'] = credit['STATUS'].apply(is_risky)

    # Aggregate credit features
    agg = credit.groupby('ID').agg(
        open_month  = ('MONTHS_BALANCE', 'min'),   # earliest month on record
        end_month   = ('MONTHS_BALANCE', 'max'),   # latest month on record
        total_loans = ('MONTHS_BALANCE', 'count'), # total loan-months
        emi_paid_off= ('STATUS', lambda s: (s == 'C').sum()),
        emi_overdue = ('STATUS', lambda s: s.isin(['1','2','3','4','5']).sum()),
        target      = ('risk', 'max'),             # 1 if ever risky
    ).reset_index()

    # window = how many months of credit history available
    agg['window'] = agg['end_month'] - agg['open_month']

    print(f"  Credit aggregation shape : {agg.shape}")
    print(f"  Target distribution:\n{agg['target'].value_counts()}")

    # ── 3.7 Merge datasets ────────────────────────────────────────────────
    merged = pd.merge(app, agg, on='ID', how='inner')
    print(f"  Merged dataset shape : {merged.shape}")

    return merged


# ===========================================================================
# STEP 4 — Label Encoding
# ===========================================================================

def encode_features(df: pd.DataFrame):
    """
    Label-encode the remaining string categorical columns.
    Returns (encoded_df, encoders_dict).
    The encoders dict is saved to encoders.pkl so the Flask app can
    apply the exact same transform to live user input.
    """
    print("\n[STEP 4] Label Encoding …")

    gender_le  = LabelEncoder()
    car_le     = LabelEncoder()
    realty_le  = LabelEncoder()

    df = df.copy()
    df['CODE_GENDER']    = gender_le.fit_transform(df['CODE_GENDER'])
    df['FLAG_OWN_CAR']   = car_le.fit_transform(df['FLAG_OWN_CAR'])
    df['FLAG_OWN_REALTY']= realty_le.fit_transform(df['FLAG_OWN_REALTY'])

    encoders = {
        'gender_le' : gender_le,
        'car_le'    : car_le,
        'realty_le' : realty_le,
    }

    print("  Encoded: CODE_GENDER, FLAG_OWN_CAR, FLAG_OWN_REALTY")
    print(f"  gender classes  : {list(gender_le.classes_)}")
    print(f"  car classes     : {list(car_le.classes_)}")
    print(f"  realty classes  : {list(realty_le.classes_)}")

    return df, encoders


# ===========================================================================
# STEP 5 — Model Training Functions
# ===========================================================================

def logistic_model(xtrain, xtest, ytrain, ytest):
    """Train and evaluate Logistic Regression."""
    print("\n[MODEL] Logistic Regression …")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(xtrain, ytrain)
    y_pred = lr_model.predict(xtest)

    cm = confusion_matrix(ytest, y_pred)
    _plot_confusion_matrix(cm, 'Logistic Regression')

    print(classification_report(ytest, y_pred))
    return lr_model, y_pred


def random_forest(X_train, X_test, y_train, y_test):
    """Train and evaluate Random Forest Classifier."""
    print("\n[MODEL] Random Forest …")
    rf_model = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    print("  Random Forest Model Evaluation")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    _plot_confusion_matrix(confusion_matrix(y_test, y_pred), 'Random Forest')

    return rf_model, y_pred


def d_tree(xtrain, xtest, ytrain, ytest):
    """Train and evaluate Decision Tree Classifier."""
    print("\n[MODEL] Decision Tree …")
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(xtrain, ytrain)
    ypred = dt.predict(xtest)

    print('  *** DecisionTreeClassifier ***')
    print('  Confusion matrix')
    print(confusion_matrix(ytest, ypred))
    print('  Classification report')
    print(classification_report(ytest, ypred))
    _plot_confusion_matrix(confusion_matrix(ytest, ypred), 'Decision Tree')

    return dt, ypred


def xgboost_model(xtrain, xtest, ytrain, ytest):
    """Train and evaluate XGBoost Classifier."""
    print("\n[MODEL] XGBoost …")
    xgb = XGBClassifier(
        eval_metric='logloss', random_state=42,
        use_label_encoder=False, verbosity=0)
    xgb.fit(xtrain, ytrain)
    ypred = xgb.predict(xtest)

    print('  *** XGBoost Classifier ***')
    print('  Confusion matrix')
    print(confusion_matrix(ytest, ypred))
    print('  Classification report')
    print(classification_report(ytest, ypred))
    _plot_confusion_matrix(confusion_matrix(ytest, ypred), 'XGBoost')

    return xgb, ypred


def _plot_confusion_matrix(cm, model_name: str):
    """Save a heatmap of the confusion matrix to PLOT_DIR."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', ax=ax,
        xticklabels=['Predicted 0', 'Predicted 1'],
        yticklabels=['True 0', 'True 1']
    )
    ax.set_title(f'{model_name} — Confusion Matrix', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    fname = os.path.join(PLOT_DIR, f'cm_{model_name.lower().replace(" ", "_")}.png')
    fig.savefig(fname, dpi=100, bbox_inches='tight')
    plt.close(fig)


# ===========================================================================
# STEP 6 — Hyperparameter Tuning (RandomizedSearchCV on best model)
# ===========================================================================

def tune_random_forest(X_train, y_train):
    """
    RandomizedSearchCV tuning for Random Forest.
    Returns the best estimator.
    """
    print("\n[TUNE] RandomizedSearchCV on Random Forest …")
    param_dist = {
        'n_estimators'      : [50, 100, 200, 300],
        'max_depth'         : [None, 5, 10, 20, 30],
        'min_samples_split' : [2, 5, 10],
        'min_samples_leaf'  : [1, 2, 4],
        'max_features'      : ['sqrt', 'log2'],
    }
    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
    search = RandomizedSearchCV(
        rf_base, param_dist,
        n_iter=20, scoring='f1', cv=3,
        random_state=42, n_jobs=-1, verbose=0
    )
    search.fit(X_train, y_train)
    print(f"  Best params : {search.best_params_}")
    print(f"  Best CV F1  : {search.best_score_:.4f}")
    return search.best_estimator_


# ===========================================================================
# MAIN — Orchestration
# ===========================================================================

def main():
    print("=" * 60)
    print("  Credit Card Approval — ML Training Pipeline")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────
    app, credit = load_data()

    # ── EDA ───────────────────────────────────────────────────────────────
    run_eda(app)

    # ── Preprocess ────────────────────────────────────────────────────────
    merged = data_cleaning(app, credit)

    # ── Encode ────────────────────────────────────────────────────────────
    merged, encoders = encode_features(merged)

    # ── Feature / target split ────────────────────────────────────────────
    drop_cols = ['ID', 'target']
    feature_cols = [c for c in merged.columns if c not in drop_cols]

    X = merged[feature_cols].fillna(0)
    y = merged['target']

    print(f"\n  Feature matrix shape : {X.shape}")
    print(f"  Target distribution  :\n{y.value_counts()}")

    # Save column order so Flask can build input correctly
    encoders['feature_cols'] = feature_cols
    encoders['housing_map']  = _HOUSING_MAP
    encoders['income_map']   = _INCOME_MAP
    encoders['education_map']= _EDUCATION_MAP
    encoders['family_map']   = _FAMILY_MAP

    # ── Train / Test split ────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    print(f"\n  Train size : {X_train.shape[0]}")
    print(f"  Test size  : {X_test.shape[0]}")

    # ── Train all 4 models ────────────────────────────────────────────────
    lr_model,  lr_pred  = logistic_model(X_train, X_test, y_train, y_test)
    rf_model,  rf_pred  = random_forest(X_train, X_test, y_train, y_test)
    dt_model,  dt_pred  = d_tree(X_train, X_test, y_train, y_test)
    xgb_model, xgb_pred = xgboost_model(X_train, X_test, y_train, y_test)

    # ── Model Comparison ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON (F1-score on test set)")
    print("=" * 60)
    results = {
        'Logistic Regression': f1_score(y_test, lr_pred,  average='weighted'),
        'Random Forest'      : f1_score(y_test, rf_pred,  average='weighted'),
        'Decision Tree'      : f1_score(y_test, dt_pred,  average='weighted'),
        'XGBoost'            : f1_score(y_test, xgb_pred, average='weighted'),
    }
    for name, score in sorted(results.items(), key=lambda x: -x[1]):
        bar = '=' * int(score * 40)
        print(f"  {name:<25} F1={score:.4f}  [{bar}]")

    best_name  = max(results, key=results.get)
    model_map  = {
        'Logistic Regression': lr_model,
        'Random Forest'      : rf_model,
        'Decision Tree'      : dt_model,
        'XGBoost'            : xgb_model,
    }
    best_model = model_map[best_name]
    print(f"\n  ✓ Best model : {best_name} (F1={results[best_name]:.4f})")

    # ── Hyperparameter tuning on best candidate ───────────────────────────
    # Always tune Random Forest as it's typically the top performer;
    # replace best_model if tuned version is better.
    tuned_rf = tune_random_forest(X_train, y_train)
    tuned_f1 = f1_score(y_test, tuned_rf.predict(X_test), average='weighted')
    print(f"  Tuned RF F1 on test : {tuned_f1:.4f}")

    if tuned_f1 >= results[best_name]:
        print("  ✓ Using tuned Random Forest as final model.")
        best_model = tuned_rf
    else:
        print(f"  ✓ Keeping original {best_name} as final model.")

    # ── Save model + encoders ─────────────────────────────────────────────
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(best_model, f)
    with open(ENCODERS_PATH, 'wb') as f:
        pickle.dump(encoders, f)

    print(f"\n  Saved model    → {MODEL_PATH}")
    print(f"  Saved encoders → {ENCODERS_PATH}")
    print("\n[DONE] Training pipeline complete.")

    # =========================================================================
    # OPTIONAL — IBM Watson Machine Learning Cloud Deployment
    # =========================================================================
    # Uncomment the block below and fill in your IBM API key + space ID.
    #
    # from ibm_watson_machine_learning import APIClient
    #
    # wml_credentials = {
    #     "url"    : "https://us-south.ml.cloud.ibm.com",
    #     "apikey" : "<YOUR_IBM_API_KEY>",
    # }
    # client = APIClient(wml_credentials)
    # client.set.default_space("<YOUR_SPACE_ID>")
    #
    # model_details = client.repository.store_model(
    #     model=best_model,
    #     meta_props={
    #         client.repository.ModelMetaNames.NAME:
    #             "credit_card_approval_model",
    #         client.repository.ModelMetaNames.TYPE:
    #             "scikit-learn_1.0",
    #         client.repository.ModelMetaNames.SOFTWARE_SPEC_UID:
    #             client.software_specifications.get_id_by_name(
    #                 "runtime-22.1-py3.9"),
    #     }
    # )
    # model_id = client.repository.get_model_id(model_details)
    #
    # deployment = client.deployments.create(
    #     model_id,
    #     meta_props={
    #         client.deployments.ConfigurationMetaNames.NAME:
    #             "credit_card_approval_deployment",
    #         client.deployments.ConfigurationMetaNames.ONLINE: {},
    #     }
    # )
    # deployment_id = client.deployments.get_id(deployment)
    # print(f"[IBM] Deployment ID: {deployment_id}")
    # =========================================================================


if __name__ == '__main__':
    main()
