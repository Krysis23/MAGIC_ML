import os
import uuid
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io,base64

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from sklearn.metrics import(
    accuracy_score, f1_score, roc_auc_score, classification_report, mean_squared_error, r2_score, mean_absolute_error
)

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


SESSIONS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'sessions')
os.makedirs(SESSIONS_DIR, exist_ok=True)


def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64

def detech_problem_type(series):
    n_unique = series.nunique()
    if series.dtype == object or n_unique <= 2:
        return 'binary' if n_unique <= 2 else 'multiclass'
    if n_unique <= 20:
        return 'multiclass'
    return 'regression'

def infer_column_types(df,target_col):
    feature_cols = [c for c in df.columns if c != target_col]
    numeric_cols = df[feature_cols].select_dtypes(include=['number']).columns.tolist()
    cat_cols = df[feature_cols].select_dtypes(exclude=['number']).columns.tolist()

    for col in list(numeric_cols):
        if df[col].nunique() <= 10 and df[col].nunique() >= 2:
            cat_cols.append(col)
            numeric_cols.remove(col)
    return numeric_cols, cat_cols


def build_preprocessor(numeric_cols,cat_cols):
    numeric_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    transformers = []
    if numeric_cols:
        transformers.append(('num', numeric_pipe, numeric_cols))
    if cat_cols:
        transformers.appen(('cat',cat_pipe,cat_cols))
    return ColumnTransformer(transformers=transformers, remainder='drop')


def get_models(problem_type):
    if problem_type in ('binary', 'multiclass'):
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        }
        if HAS_XGB:
            models['XGBoost'] = XGBClassifier(
                n_estimators=100, random_state=42,
                eval_metrics='logloss', verbosity=0
            )
        else:
            models= {
                'Rigde Regression': Ridge(),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
            }
            if HAS_XGB:
                models['XGBoost'] = XGBRegressor(
                    n_estimators=100, random_state=42, verbosity=0
                )
        return models
    

def evaluate_model(model, X_test, y_test, problem_type):
    y_pred = model.predict(X_test)
    if problem_type == 'regression':
        rmse = float(np.sqrt(mean_squared_error(y_test,y_pred)))
        return {
            'R2': round(float(r2_score(y_test,y_pred)), 4),
            'MAE': round(float(mean_absolute_error(y_test,y_pred)), 4),
            'RMSE': round(rmse, 4)
        }
    else:
        avg = 'binary' if problem_type == 'binary' else 'weighted'
        metrics = {
            'Accuracy': round(float(accuracy_score(y_test, y_pred)), 4),
            'F1': round(float(f1_score(y_test, y_pred, average=avg, zero_division=0)), 4),
        }
        try:
            if problem_type == 'binary':
                proba = model.predict_proba(X_test)[:,1]
                metrics['ROC-AUC'] = round(float(roc_auc_score(y_test, proba)), 4)
        except Exception:
            pass
        return metrics
    

def generate_eda_plots(df, target_col):
    plots = {}

    if df.isnull().sum().sum() > 0:
        fig, ax = plt.subplots(figsize=(8,4))
        missing = df.isnull().mean().sort_values(ascending=False)
        missing = missing[missing > 0]
        sns.barplot(x=missing.values, y=missing.index, ax=ax, color='#4f86c6')
        ax.set_title('Missing Value Rate by Column', fontsize=12)
        ax.set_xlable('Missing Rate')
        plots['missing'] = _fig_to_b64(fig)

    fig, ax = plt.subplots(figsize=(6,4))
    if df[target_col].dtype == object or df[target_col].nunique() <= 20:
        df[target_col].value_counts().plot(kind='bar', ax=ax, color='#4f86c6', edgecolor='white')
        ax.set_title('Target Distribution', fontsize=12)
        ax.set_xlabel(target_col)
        plt.xticks(rotation=45)
    else:
        ax.hist(df[target_col].dropna(), bins=30, color='#4f86c6',edgecolor='white')
        ax.set_title('Target Distribution', fontsize=12)
        ax.set_xlabel(target_col)
    plots['target'] = _fig_to_b64(fig)

    num_df = df.select_dtypes(include='number')
    if len(num_df.columns) >= 2:
        fig, ax = plt.subplots(figsize=(8,6))
        corr = num_df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=len(corr) <= 12, fmt='.2f', cmap='coolwarm', ax=ax, linewidths=0.5, vmin=-1, vmax=1)
    ax.set_title('Correlation Matrix', fontsize=12)
    plots['correlation'] = _fig_to_b64(fig)

    return plots

