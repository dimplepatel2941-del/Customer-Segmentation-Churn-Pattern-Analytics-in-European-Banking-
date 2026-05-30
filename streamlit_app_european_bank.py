import pathlib
import pandas as pd
import streamlit as st
import altair as alt
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_DIR = pathlib.Path(__file__).resolve().parent
DATA_FILE = DATA_DIR / "European_Bank.csv"
MODEL_FILE = DATA_DIR / "european_bank_rf.joblib"

st.set_page_config(page_title="European Bank Churn Analytics", layout="wide")

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

@st.cache_resource
def build_model(df):
    features = [c for c in df.columns if c not in ["Exited", "CustomerId", "Year"]]
    processed = pd.get_dummies(df[features], drop_first=True)
    y = df["Exited"].astype(int)
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(processed, y)
    return model, processed.columns.tolist()

@st.cache_resource
def load_or_train(df, model_path):
    if model_path.exists():
        try:
            model, feature_columns = joblib.load(model_path)
            return model, feature_columns
        except Exception:
            pass
    model, feature_columns = build_model(df)
    try:
        joblib.dump((model, feature_columns), model_path)
    except Exception:
        pass
    return model, feature_columns


def prepare_feature_row(inputs, feature_columns):
    row = pd.DataFrame([inputs])
    row = pd.get_dummies(row, drop_first=True)
    for col in feature_columns:
        if col not in row.columns:
            row[col] = 0
    return row[feature_columns]


def main():
    st.title("Customer Segmentation & Churn Pattern Analytics")
    st.markdown(
        "Analyze European banking churn patterns and generate quick customer churn risk predictions from the supplied project dataset."
    )

    df = load_data(DATA_FILE)
    model, feature_columns = load_or_train(df, MODEL_FILE)

    st.sidebar.header("Prediction Input")
    geography = st.sidebar.selectbox("Geography", sorted(df["Geography"].unique()), index=0)
    gender = st.sidebar.selectbox("Gender", sorted(df["Gender"].unique()), index=0)
    credit_score = st.sidebar.slider("Credit Score", int(df["CreditScore"].min()), int(df["CreditScore"].max()), int(df["CreditScore"].median()))
    age = st.sidebar.slider("Age", int(df["Age"].min()), int(df["Age"].max()), int(df["Age"].median()))
    tenure = st.sidebar.slider("Tenure", int(df["Tenure"].min()), int(df["Tenure"].max()), int(df["Tenure"].median()))
    balance = st.sidebar.number_input("Balance", min_value=0.0, value=float(df["Balance"].median()), step=100.0, format="%.2f")
    num_products = st.sidebar.selectbox("Number of Products", sorted(df["NumOfProducts"].unique()))
    has_credit_card = st.sidebar.selectbox("Has Credit Card", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    is_active = st.sidebar.selectbox("Is Active Member", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    estimated_salary = st.sidebar.number_input("Estimated Salary", min_value=0.0, value=float(df["EstimatedSalary"].median()), step=1000.0, format="%.2f")

    input_features = {
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_credit_card,
        "IsActiveMember": is_active,
        "EstimatedSalary": estimated_salary,
        "Geography": geography,
        "Gender": gender,
    }

    row = prepare_feature_row(input_features, feature_columns)
    prediction = model.predict(row)[0]
    probability = model.predict_proba(row)[0][1]

    st.subheader("Quick Churn Risk Score")
    st.metric("Predicted Churn", "Yes" if prediction == 1 else "No", f"{probability:.1%} probability")

    with st.expander("View Selected Input Features"):
        st.json(input_features)

    st.header("Dataset Overview")
    st.write(df.head(10))

    churn_rate = df["Exited"].mean()
    st.metric("Overall Churn Rate", f"{churn_rate:.1%}")

    col1, col2 = st.columns(2)
    with col1:
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X("Geography:N"), y=alt.Y("count():Q"), color="Exited:N"
        )
        st.altair_chart(chart, use_container_width=True)
    with col2:
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X("Gender:N"), y=alt.Y("count():Q"), color="Exited:N"
        )
        st.altair_chart(chart, use_container_width=True)

    st.header("Model Performance")
    features = [c for c in df.columns if c not in ["Exited", "CustomerId", "Year"]]
    processed = pd.get_dummies(df[features], drop_first=True)
    y_true = df["Exited"].astype(int)
    y_pred = model.predict(processed[feature_columns])
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    confusion = confusion_matrix(y_true, y_pred)

    st.write(f"**Accuracy:** {accuracy:.3f}")
    st.write("**Classification Report:**")
    st.dataframe(pd.DataFrame(report).transpose())
    st.write("**Confusion Matrix:**")
    st.write(confusion)

    st.caption("App generated from European Bank churn dataset in the Customer Segmentation & Churn Pattern Analytics project.")


if __name__ == "__main__":
    main()
