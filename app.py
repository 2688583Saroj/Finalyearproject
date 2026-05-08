import json
import os
import pickle
import hashlib
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Crop Yield Prediction System",
    page_icon="Crop",
    layout="wide",
)

DATASETS = {
    "India Crop Yield Dataset": {
        "key": "india_crop_yield",
        "csv": "data/crop_yield.csv",
        "target": "Yield",
        "description": "Predicts crop yield using crop, year, season, state, area,"
        " production, rainfall, fertilizer and pesticide.",
    },
    "Soil + Weather Dataset": {
        "key": "soil_weather_yield",
        "csv": "data/crop-yield.csv",
        "target": "Crop_Yield_ton_per_hectare",
        "description": "Predicts crop yield using soil nutrients, climate, "
        "region, crop type, irrigation, fertilizer and pesticide.",
    },
}

RECORDS_FILE = "data/app_records.json"
ADMIN_EMAIL = "saroj@2688583.cropyield"
ADMIN_PASSWORD = "Uel@CN6000"


def apply_custom_css():
    st.markdown(
        """
        <style>
        :root {
            --bg: #f4f7f1;
            --surface: #ffffff;
            --surface-soft: #eef6ea;
            --text: #203528;
            --muted: #60705f;
            --accent: #2f7d4f;
            --accent-dark: #1f5e39;
            --line: #d9e4d5;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(129, 184, 121, 0.24), transparent 34rem),
                linear-gradient(135deg, #f7faf5 0%, var(--bg) 48%, #edf4e9 100%);
            color: var(--text);
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #193523 0%, #244d31 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.12);
        }

        [data-testid="stSidebar"] * {
            color: #f4fff4;
        }

        [data-testid="stSidebar"] .stSelectbox label {
            color: #e9f7e9;
            font-weight: 700;
        }

        [data-testid="stSidebar"] [data-baseweb="select"] > div {
            background: rgba(255, 255, 255, 0.12);
            border: 1px solid rgba(255, 255, 255, 0.24);
            border-radius: 8px;
        }

        .hero {
            background:
                linear-gradient(135deg, rgba(31, 94, 57, 0.94), rgba(47, 125, 79, 0.84)),
                url("https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1600&q=80");
            background-size: cover;
            background-position: center;
            border-radius: 8px;
            padding: 2.4rem 2.2rem;
            margin-bottom: 1.3rem;
            box-shadow: 0 22px 55px rgba(29, 63, 39, 0.18);
        }

        .hero h1 {
            color: #ffffff;
            font-size: clamp(2rem, 4vw, 3.4rem);
            line-height: 1.04;
            margin: 0 0 0.75rem;
            letter-spacing: 0;
        }

        .hero p {
            color: #effaf0;
            max-width: 720px;
            font-size: 1.06rem;
            margin: 0;
        }

        .section-card {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1.15rem 1.25rem;
            margin: 0.9rem 0 1.1rem;
            box-shadow: 0 12px 32px rgba(44, 75, 50, 0.08);
        }

        h2, h3 {
            color: var(--text);
            letter-spacing: 0;
        }

        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, #ffffff 0%, #f4faf2 100%);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 10px 24px rgba(46, 84, 51, 0.08);
        }

        div[data-testid="stMetricLabel"] p {
            color: var(--muted);
            font-weight: 700;
        }

        div[data-testid="stMetricValue"] {
            color: var(--accent-dark);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.45rem;
            border-bottom: 1px solid var(--line);
        }

        .stTabs [data-baseweb="tab"] {
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid var(--line);
            border-bottom: 0;
            border-radius: 8px 8px 0 0;
            padding: 0.65rem 1rem;
            font-weight: 700;
        }

        .stTabs [aria-selected="true"] {
            background: #ffffff;
            color: var(--accent-dark);
        }

        .stButton > button {
            background: linear-gradient(135deg, var(--accent), var(--accent-dark));
            color: #ffffff;
            border: 0;
            border-radius: 8px;
            padding: 0.72rem 1.1rem;
            font-weight: 800;
            box-shadow: 0 12px 24px rgba(47, 125, 79, 0.24);
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, #398d5d, #1b5432);
            color: #ffffff;
            border: 0;
        }

        [data-testid="stNumberInput"] input,
        [data-baseweb="select"] > div {
            border-radius: 8px;
            border-color: var(--line);
            background: #ffffff;
        }

        [data-testid="stDataFrame"],
        [data-testid="stTable"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 10px 28px rgba(44, 75, 50, 0.07);
        }

        [data-testid="stAlert"] {
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    st.markdown(
        """
        <div class="hero">
            <h1>Crop Yield Prediction System</h1>
            <p>
                Random Forest Regression dashboard for crop-yield prediction,
                evaluation, and decision support.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def init_session_state():
    defaults = {
        "role": "guest",
        "user": None,
        "page": "Home",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def password_hash(password):
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def load_records():
    if not os.path.exists(RECORDS_FILE):
        return {"users": [], "predictions": []}

    with open(RECORDS_FILE, "r") as f:
        return json.load(f)


def save_records(records):
    os.makedirs(os.path.dirname(RECORDS_FILE), exist_ok=True)
    with open(RECORDS_FILE, "w") as f:
        json.dump(records, f, indent=4)


def find_user(records, email):
    email = email.strip().lower()
    return next((user for user in records["users"] if user["email"] == email), None)


def register_user(name, email, password, location):
    records = load_records()
    email = email.strip().lower()

    if find_user(records, email):
        return False, "An account with this email already exists."

    records["users"].append(
        {
            "name": name.strip(),
            "email": email,
            "password_hash": password_hash(password),
            "location": location.strip(),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    save_records(records)
    return True, "Registration successful. You can now log in."


def login_user(email, password):
    email = email.strip().lower()

    if email == ADMIN_EMAIL and password == ADMIN_PASSWORD:
        st.session_state.role = "admin"
        st.session_state.user = {"name": "Administrator", "email": ADMIN_EMAIL}
        st.session_state.page = "Admin"
        st.session_state.login_error = ""
        return True

    records = load_records()
    user = find_user(records, email)
    if user and user["password_hash"] == password_hash(password):
        st.session_state.role = "user"
        st.session_state.user = {
            "name": user["name"],
            "email": user["email"],
            "location": user.get("location", ""),
        }
        st.session_state.page = "Predict"
        st.session_state.login_error = ""
        return True

    return False


def logout():
    st.session_state.role = "guest"
    st.session_state.user = None
    st.session_state.page = "Home"
    st.rerun()


def save_prediction(user, dataset_label, input_data, prediction):
    records = load_records()
    records["predictions"].append(
        {
            "user_email": user["email"],
            "user_name": user["name"],
            "dataset": dataset_label,
            "prediction": round(float(prediction), 4),
            "inputs": input_data,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    save_records(records)


def load_model_and_metrics(dataset_key):
    model_path = f"model/{dataset_key}_model.pkl"
    metrics_path = f"model/{dataset_key}_metrics.json"

    if not os.path.exists(model_path) or not os.path.exists(metrics_path):
        return None, None

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    return model, metrics


def get_default_value(series):
    if pd.api.types.is_numeric_dtype(series):
        return float(series.median())
    return str(series.mode().iloc[0]) if not series.mode().empty else str(series.dropna().iloc[0])


def prediction_form(df, target, model, dataset_label):
    features = [c for c in df.columns if c != target]
    input_data = {}

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Enter Crop Details")

    cols = st.columns(2)

    for index, column in enumerate(features):
        series = df[column].dropna()
        with cols[index % 2]:
            if pd.api.types.is_numeric_dtype(series):
                min_val = float(series.min())
                max_val = float(series.max())
                default_val = get_default_value(series)

                input_data[column] = st.number_input(
                    column,
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                )
            else:
                options = sorted(series.astype(str).unique().tolist())
                input_data[column] = st.selectbox(column, options)

    if st.button("Predict Yield", type="primary"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        save_prediction(st.session_state.user, dataset_label, input_data, prediction)

        st.success(f"Predicted Crop Yield: {prediction:.4f}")
        st.info("This prediction has been saved to your account history.")

    st.markdown("</div>", unsafe_allow_html=True)


def show_dashboard(df, target, metrics):
    st.subheader("Model Evaluation")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows Used", metrics["rows"])
    c2.metric("R2 Score", metrics["r2_score"])
    c3.metric("MAE", metrics["mae"])
    c4.metric("RMSE", metrics["rmse"])

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if target in numeric_cols:
        st.subheader("Yield Distribution")
        fig, ax = plt.subplots()
        ax.hist(df[target].dropna(), bins=30, color="#2f7d4f", edgecolor="#f7faf5")
        ax.set_xlabel(target)
        ax.set_ylabel("Frequency")
        ax.set_facecolor("#f7faf5")
        fig.patch.set_facecolor("#f7faf5")
        st.pyplot(fig)

    st.subheader("Correlation with Yield")
    corr = df[numeric_cols].corr(numeric_only=True)[target].sort_values(ascending=False)
    st.dataframe(corr.to_frame("Correlation"), use_container_width=True)

    st.subheader("Report Support")
    st.write(
        "Use these results in your dissertation evaluation section: R2 shows prediction strength, "
        "MAE shows average error, and RMSE penalises larger prediction errors."
    )


def show_home():
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Home")
    st.write(
        "This crop yield prediction system allows registered users to enter crop, "
        "soil, weather and farm details, then receive Random Forest based yield predictions."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Datasets", len(DATASETS))
    c2.metric("Models", "Random Forest")
    c3.metric("Access", "User + Admin")

    st.write(
        "Users can register, log in, predict crop yield, and keep a prediction history. "
        "The admin can review users, predictions, datasets, and model evaluation results."
    )
    st.markdown("</div>", unsafe_allow_html=True)


def show_auth_page():
    login_tab, register_tab = st.tabs(["Login", "Register"])

    with login_tab:
        st.subheader("Login")
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login", type="primary"):
            if login_user(email, password):
                st.success("Login successful.")
                st.rerun()
            else:
                st.session_state.login_error = (
                    "Wrong email or password. Please check your login details and try again."
                )

        if st.session_state.get("login_error"):
            st.error(st.session_state.login_error)
            st.warning("Login failed. Use your registered user account or the admin credentials.")

        st.info("Admin login: saroj@2688583.cropyield / Uel@CN6000")

    with register_tab:
        st.subheader("Create User Account")
        name = st.text_input("Full Name")
        email = st.text_input("Email Address")
        location = st.text_input("Farm Location")
        password = st.text_input("Create Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")

        if st.button("Register", type="primary"):
            if not name or not email or not password:
                st.error("Please enter name, email and password.")
            elif password != confirm_password:
                st.error("Passwords do not match.")
            else:
                success, message = register_user(name, email, password, location)
                if success:
                    st.success(message)
                else:
                    st.error(message)


def show_prediction_page():
    if st.session_state.role != "user":
        st.warning("Please log in as a registered user to predict crop yield.")
        show_auth_page()
        return

    dataset_label = st.sidebar.selectbox("Select Dataset", list(DATASETS.keys()))
    config = DATASETS[dataset_label]
    st.sidebar.write(config["description"])

    if not os.path.exists(config["csv"]):
        st.error(f"Dataset not found: {config['csv']}")
        return

    df = pd.read_csv(config["csv"])
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates()

    model, metrics = load_model_and_metrics(config["key"])

    if model is None:
        st.warning("Model is not trained yet.")
        st.code("python train_model.py\npython -m streamlit run app.py")
        return

    st.subheader(f"Welcome, {st.session_state.user['name']}")
    prediction_form(df, config["target"], model, dataset_label)

    records = load_records()
    user_predictions = [
        item for item in records["predictions"]
        if item["user_email"] == st.session_state.user["email"]
    ]
    if user_predictions:
        st.subheader("Your Prediction History")
        history = pd.DataFrame(user_predictions)
        st.dataframe(
            history[["created_at", "dataset", "prediction"]],
            use_container_width=True,
        )


def show_analytics_page():
    dataset_label = st.sidebar.selectbox(
        "Select Analytics Dataset",
        list(DATASETS.keys()),
    )
    config = DATASETS[dataset_label]
    st.sidebar.write(config["description"])

    if not os.path.exists(config["csv"]):
        st.error(f"Dataset not found: {config['csv']}")
        return

    df = pd.read_csv(config["csv"])
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates()

    model, metrics = load_model_and_metrics(config["key"])
    if metrics is None:
        st.warning("Model metrics not found. Run training first.")
        st.code("python train_model.py")
        return

    st.subheader(dataset_label)
    show_dashboard(df, config["target"], metrics)


def show_admin_page():
    if st.session_state.role != "admin":
        st.warning("Admin access required.")
        show_auth_page()
        return

    records = load_records()
    users = pd.DataFrame(records["users"])
    predictions = pd.DataFrame(records["predictions"])

    st.subheader("Admin Dashboard")

    c1, c2, c3 = st.columns(3)
    c1.metric("Registered Users", len(records["users"]))
    c2.metric("Predictions", len(records["predictions"]))
    c3.metric("Datasets", len(DATASETS))

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Users", "Predictions", "Model Evaluation", "Datasets"]
    )

    with tab1:
        st.subheader("Registered Users")
        if users.empty:
            st.info("No users registered yet.")
        else:
            st.dataframe(
                users.drop(columns=["password_hash"], errors="ignore"),
                use_container_width=True,
            )

    with tab2:
        st.subheader("All User Predictions")
        if predictions.empty:
            st.info("No predictions saved yet.")
        else:
            st.dataframe(
                predictions[["created_at", "user_name", "user_email", "dataset", "prediction"]],
                use_container_width=True,
            )

    with tab3:
        dataset_label = st.selectbox("Select Model Dataset", list(DATASETS.keys()))
        config = DATASETS[dataset_label]
        if os.path.exists(config["csv"]):
            df = pd.read_csv(config["csv"])
            df.columns = df.columns.str.strip()
            df = df.drop_duplicates()
            model, metrics = load_model_and_metrics(config["key"])
            if metrics is None:
                st.warning("Model metrics not found.")
            else:
                show_dashboard(df, config["target"], metrics)
        else:
            st.error(f"Dataset not found: {config['csv']}")

    with tab4:
        dataset_label = st.selectbox("Inspect Dataset", list(DATASETS.keys()))
        config = DATASETS[dataset_label]
        if os.path.exists(config["csv"]):
            df = pd.read_csv(config["csv"])
            st.write(config["description"])
            st.dataframe(df.head(30), use_container_width=True)
        else:
            st.error(f"Dataset not found: {config['csv']}")


def show_project_info():
    st.subheader("About This Project")
    st.write(
        """
        This system predicts crop yield using Random Forest Regression.
        It follows the project methodology: data collection, preprocessing,
        model training, model evaluation, prediction and decision support.
        """
    )
    st.write(
        """
        The system supports two datasets:
        1. Indian crop yield dataset.
        2. Soil and weather based crop yield dataset.
        """
    )


def main():
    apply_custom_css()
    init_session_state()
    render_header()

    st.sidebar.title("Navigation")
    pages = ["Home", "Login / Register", "Project Info"]
    if st.session_state.role == "user":
        pages.insert(1, "Predict")
    if st.session_state.role == "admin":
        pages.insert(1, "Admin")
    if st.session_state.role in ["user", "admin"]:
        pages.insert(-1, "Analytics")

    page = st.sidebar.radio("Go to", pages, index=pages.index(st.session_state.page)
                            if st.session_state.page in pages else 0)
    st.session_state.page = page

    if st.session_state.user:
        st.sidebar.write(f"Signed in as {st.session_state.user['name']}")
        if st.sidebar.button("Logout"):
            logout()

    if page == "Home":
        show_home()
    elif page == "Login / Register":
        show_auth_page()
    elif page == "Predict":
        show_prediction_page()
    elif page == "Admin":
        show_admin_page()
    elif page == "Analytics":
        show_analytics_page()
    elif page == "Project Info":
        show_project_info()


if __name__ == "__main__":
    main()
