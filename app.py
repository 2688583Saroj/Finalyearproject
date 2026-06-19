import json
import os
import pickle
import hashlib
import hmac
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
        "drop_columns": [],
        "description": "Predicts crop yield using crop, year, season, state, area,"
        " production, rainfall, fertilizer and pesticide.",
    }
}

RECORDS_FILE = "data/app_records.json"
PASSWORD_HASH_ITERATIONS = 260000


def get_setting(name, default=""):
    env_value = os.getenv(name)
    if env_value:
        return env_value

    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


ADMIN_EMAIL = get_setting("ADMIN_EMAIL")
ADMIN_PASSWORD = get_setting("ADMIN_PASSWORD")


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

        [data-testid="stSidebar"] .stButton > button {
            width: 100%;
            justify-content: flex-start;
            margin-bottom: 0.35rem;
            box-shadow: none;
            transition: transform 0.15s ease, background 0.15s ease;
        }

        [data-testid="stSidebar"] .stButton > button:hover {
            transform: translateX(4px);
        }

        [data-testid="stSidebar"] .stButton > button[kind="primary"] {
            background: rgba(255, 255, 255, 0.16);
            border: 1px solid rgba(255, 255, 255, 0.34);
            box-shadow: inset 4px 0 0 #9be27f;
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
    salt = os.urandom(16)
    hash_bytes = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_HASH_ITERATIONS,
    )
    return (
        f"pbkdf2_sha256${PASSWORD_HASH_ITERATIONS}$"
        f"{salt.hex()}${hash_bytes.hex()}"
    )


def verify_password(password, stored_hash):
    if not stored_hash:
        return False

    if stored_hash.startswith("pbkdf2_sha256$"):
        try:
            _, iterations, salt_hex, expected_hex = stored_hash.split("$")
            actual_hash = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode("utf-8"),
                bytes.fromhex(salt_hex),
                int(iterations),
            ).hex()
            return hmac.compare_digest(actual_hash, expected_hex)
        except (ValueError, TypeError):
            return False

    legacy_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()
    return hmac.compare_digest(legacy_hash, stored_hash)


def load_records():
    if not os.path.exists(RECORDS_FILE):
        return {"users": [], "predictions": []}

    try:
        with open(RECORDS_FILE, "r") as f:
            records = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"users": [], "predictions": []}

    records.setdefault("users", [])
    records.setdefault("predictions", [])
    return records


def save_records(records):
    os.makedirs(os.path.dirname(RECORDS_FILE), exist_ok=True)
    with open(RECORDS_FILE, "w") as f:
        json.dump(records, f, indent=4)


def find_user(records, email):
    email = email.strip().lower()
    return next((user for user in records["users"] if user["email"] == email), None)


def get_dataset_locations():
    primary_config = next(iter(DATASETS.values()))
    if not os.path.exists(primary_config["csv"]):
        return []

    try:
        df = prepare_dataset(primary_config)
    except (OSError, pd.errors.ParserError):
        return []

    if "State" not in df.columns:
        return []

    return sorted(df["State"].dropna().astype(str).str.strip().unique().tolist())


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

    if ADMIN_EMAIL and ADMIN_PASSWORD and email == ADMIN_EMAIL and password == ADMIN_PASSWORD:
        st.session_state.role = "admin"
        st.session_state.user = {"name": "Administrator", "email": ADMIN_EMAIL}
        st.session_state.page = "Admin"
        st.session_state.login_error = ""
        return True

    records = load_records()
    user = find_user(records, email)
    if user and verify_password(password, user.get("password_hash", "")):
        if not user.get("password_hash", "").startswith("pbkdf2_sha256$"):
            user["password_hash"] = password_hash(password)
            save_records(records)

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


def save_prediction(user, dataset_label, input_data, prediction, classification=None):
    records = load_records()
    records["predictions"].append(
        {
            "user_email": user["email"],
            "user_name": user["name"],
            "dataset": dataset_label,
            "prediction": round(float(prediction), 4),
            "classification": classification,
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


def get_feature_importance(model):
    try:
        preprocessor = model.named_steps["preprocessor"]
        regressor = model.named_steps["model"]
        feature_names = preprocessor.get_feature_names_out()
        importances = regressor.feature_importances_
    except (AttributeError, KeyError):
        return pd.DataFrame()

    if len(feature_names) != len(importances):
        return pd.DataFrame()

    importance_df = pd.DataFrame(
        {
            "Feature": [
                name.replace("num__", "").replace("cat__", "")
                for name in feature_names
            ],
            "Importance": importances,
        }
    )
    importance_df["Feature"] = importance_df["Feature"].str.replace("_", " ", regex=False)
    return importance_df.sort_values("Importance", ascending=False).head(15)


def dataframe_to_csv(dataframe):
    return dataframe.to_csv(index=False).encode("utf-8")


def get_default_value(series):
    if pd.api.types.is_numeric_dtype(series):
        return float(series.median())
    return str(series.mode().iloc[0]) if not series.mode().empty else str(series.dropna().iloc[0])


def add_engineered_features(df):
    df = df.copy()

    if {"Production", "Area"}.issubset(df.columns):
        safe_area = df["Area"].replace(0, pd.NA)
        df["Production_per_Area"] = df["Production"] / safe_area

    if {"Fertilizer", "Area"}.issubset(df.columns):
        safe_area = df["Area"].replace(0, pd.NA)
        df["Fertilizer_per_Area"] = df["Fertilizer"] / safe_area

    if {"Pesticide", "Area"}.issubset(df.columns):
        safe_area = df["Area"].replace(0, pd.NA)
        df["Pesticide_per_Area"] = df["Pesticide"] / safe_area

    return df


def prepare_dataset(config):
    df = pd.read_csv(config["csv"])
    df.columns = df.columns.str.strip()
    drop_columns = [c for c in config.get("drop_columns", []) if c in df.columns]
    df = df.drop(columns=drop_columns)
    return df.drop_duplicates()


def classify_prediction(prediction, target_series):
    low_limit = target_series.quantile(0.33)
    high_limit = target_series.quantile(0.66)

    if prediction <= low_limit:
        return "Low Yield"
    if prediction <= high_limit:
        return "Medium Yield"
    return "High Yield"


def prediction_form(df, target, model, dataset_label):
    features = [c for c in df.columns if c != target]
    input_data = {}
    user_location = str(st.session_state.user.get("location", "")).strip()

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
                    help=f"Allowed dataset range: {min_val:.2f} to {max_val:.2f}",
                )
            else:
                options = sorted(series.astype(str).unique().tolist())
                default_index = 0
                if column == "State" and user_location in options:
                    default_index = options.index(user_location)
                input_data[column] = st.selectbox(column, options, index=default_index)

    if st.button("Predict Yield", type="primary"):
        try:
            input_df = add_engineered_features(pd.DataFrame([input_data]))
            prediction = model.predict(input_df)[0]
            classification = classify_prediction(prediction, df[target].dropna())
            save_prediction(
                st.session_state.user,
                dataset_label,
                input_data,
                prediction,
                classification,
            )

            st.success(f"Predicted Crop Yield: {prediction:.4f}")
            st.info(f"Prediction Classification: {classification}")
            st.info("This prediction has been saved to your account history.")
        except (ValueError, TypeError, KeyError) as error:
            st.error("Prediction could not be completed. Please check the entered crop details.")
            st.caption(str(error))

    st.markdown("</div>", unsafe_allow_html=True)


def show_dashboard(df, target, metrics, model=None):
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

    if model is not None:
        importance_df = get_feature_importance(model)
        if not importance_df.empty:
            st.subheader("Model Feature Importance")
            st.write(
                "This chart shows which input features the Random Forest model uses most "
                "when predicting crop yield. A higher importance score means the feature "
                "had a stronger overall influence on the model's predictions."
            )
            chart_data = importance_df.set_index("Feature")
            st.bar_chart(chart_data)
            st.dataframe(importance_df, use_container_width=True)


def show_home():
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Crop Yield Decision Support")
    st.write(
        "This system uses a trained Random Forest Regression model to estimate crop "
        "yield from historical agricultural data. It combines prediction, model "
        "evaluation, user history and admin review in one Streamlit dashboard."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Datasets", len(DATASETS))
    c2.metric("Models", "Random Forest")
    c3.metric("Prediction Target", "Yield")
    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("For Registered Users")
        st.write(
            "Users can register with a dataset-supported farm location, enter crop "
            "and farming details, generate a yield prediction, view the yield category, "
            "and download their prediction history."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("For Admin Review")
        st.write(
            "The admin dashboard provides access to registered users, saved predictions, "
            "dataset previews, evaluation metrics, yield distribution, correlation with "
            "yield and model feature importance."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("How The System Works")
    step1, step2, step3, step4 = st.columns(4)
    step1.metric("1", "Register")
    step2.metric("2", "Enter Data")
    step3.metric("3", "Predict")
    step4.metric("4", "Review")
    st.write(
        "The workflow follows the machine-learning project process: dataset preparation, "
        "model training, prediction, evaluation and decision support."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Model Insights Available")
    insight1, insight2, insight3 = st.columns(3)
    with insight1:
        st.write("**Yield Distribution**")
        st.write("Shows how yield values are spread across the dataset.")
    with insight2:
        st.write("**Correlation With Yield**")
        st.write("Shows relationships between numeric inputs and yield.")
    with insight3:
        st.write("**Feature Importance**")
        st.write("Shows which inputs influence the Random Forest model most.")
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

        if ADMIN_EMAIL and ADMIN_PASSWORD:
            st.info("Admin access is configured for this deployment.")
        else:
            st.info("Admin access is configured with ADMIN_EMAIL and ADMIN_PASSWORD secrets.")

    with register_tab:
        st.subheader("Create User Account")
        name = st.text_input("Full Name")
        email = st.text_input("Email Address")
        locations = get_dataset_locations()
        if locations:
            location = st.selectbox("Farm Location", locations)
        else:
            location = st.text_input("Farm Location")
        password = st.text_input("Create Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")

        if st.button("Register", type="primary"):
            if not name or not email or not password:
                st.error("Please enter name, email and password.")
            elif locations and location not in locations:
                st.error("Please choose a farm location from the dataset.")
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

    df = prepare_dataset(config)

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
        if "classification" not in history.columns:
            history["classification"] = ""
        history_view = history[["created_at", "dataset", "prediction", "classification"]]
        st.dataframe(
            history_view,
            use_container_width=True,
        )
        st.download_button(
            "Download My Prediction History",
            data=dataframe_to_csv(history_view),
            file_name="my_prediction_history.csv",
            mime="text/csv",
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

    df = prepare_dataset(config)

    model, metrics = load_model_and_metrics(config["key"])
    if metrics is None:
        st.warning("Model metrics not found. Run training first.")
        st.code("python train_model.py")
        return

    st.subheader(dataset_label)
    show_dashboard(df, config["target"], metrics, model)


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
            if "classification" not in predictions.columns:
                predictions["classification"] = ""
            predictions_view = predictions[[
                    "created_at",
                    "user_name",
                    "user_email",
                    "dataset",
                    "prediction",
                    "classification",
                ]]
            st.dataframe(
                predictions_view,
                use_container_width=True,
            )
            st.download_button(
                "Download All Predictions",
                data=dataframe_to_csv(predictions_view),
                file_name="all_user_predictions.csv",
                mime="text/csv",
            )

    with tab3:
        dataset_label = st.selectbox("Select Model Dataset", list(DATASETS.keys()))
        config = DATASETS[dataset_label]
        if os.path.exists(config["csv"]):
            df = prepare_dataset(config)
            model, metrics = load_model_and_metrics(config["key"])
            if metrics is None:
                st.warning("Model metrics not found.")
            else:
                show_dashboard(df, config["target"], metrics, model)
        else:
            st.error(f"Dataset not found: {config['csv']}")

    with tab4:
        dataset_label = st.selectbox("Inspect Dataset", list(DATASETS.keys()))
        config = DATASETS[dataset_label]
        if os.path.exists(config["csv"]):
            df = prepare_dataset(config)
            st.write(config["description"])
            st.dataframe(df.head(30), use_container_width=True)
        else:
            st.error(f"Dataset not found: {config['csv']}")


def show_project_info():
    st.subheader("Project Overview")
    st.write(
        "This crop yield prediction system is a machine-learning decision-support "
        "application built with Random Forest Regression. It allows registered users to enter crop and farming details, generate a predicted "
        "yield value, and keep a record of their prediction history."
    )

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Aim")
    st.write(
        "The aim of this project is to predict agricultural crop yield from historical "
        "crop, seasonal, location, rainfall, fertilizer, pesticide, area and production "
        "data. The system supports farmers, students and administrators by presenting "
        "both predictions and model evaluation results in a simple dashboard."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Dataset")
    st.write("The system currently supports the Indian crop yield dataset.")
    st.write(
        "The main input variables are Crop, Crop Year, Season, State, Area, Production, "
        "Annual Rainfall, Fertilizer and Pesticide. The target variable predicted by "
        "the model is Yield."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Methodology")
    st.write(
        "The dataset is cleaned by removing duplicate records and separating the target "
        "variable from the input features. Numeric values are imputed with median values, "
        "categorical values are encoded, and engineered features such as production per "
        "area, fertilizer per area and pesticide per area are added before training."
    )
    st.write(
        "A Random Forest Regression model is trained because it handles both linear and "
        "non-linear relationships, works well with mixed feature types, and can provide "
        "feature importance values for interpretation."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Evaluation")
    st.write(
        "The model is evaluated using R2 Score, Mean Absolute Error and Root Mean Squared "
        "Error. The analytics dashboard also includes yield distribution, correlation "
        "with yield, and model feature importance to explain the dataset and model behavior."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Model Limitations")
    st.write(
        "The prediction is based on historical dataset patterns, so it should be treated "
        "as an estimate rather than a guaranteed result. The model may not fully capture "
        "future climate changes, extreme weather events, soil quality, irrigation quality, "
        "market changes or farming practices that are not included in the dataset."
    )
    st.write(
        "The system is decision support, but it should not replace expert agricultural advice or field-level assessment."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Future Improvements")
    st.write(
        "Future versions could use a database instead of a local JSON file, add more "
        "regional datasets, include soil and irrigation features, compare multiple "
        "machine-learning algorithms, and provide more detailed prediction explanations."
    )
    st.markdown("</div>", unsafe_allow_html=True)


def main():
    apply_custom_css()
    init_session_state()
    render_header()

    st.sidebar.title("Navigation")

    if st.session_state.role == "admin":
        pages = ["Home", "Admin", "Analytics", "Project Info"]
    elif st.session_state.role == "user":
        pages = ["Home", "Predict", "Analytics", "Project Info"]
    else:
        pages = ["Home", "Login / Register", "Project Info"]

    if st.session_state.page not in pages:
        st.session_state.page = pages[0]

    st.sidebar.caption("Menu")
    for page_name in pages:
        is_current_page = st.session_state.page == page_name
        if st.sidebar.button(
            page_name,
            key=f"nav_{page_name}",
            use_container_width=True,
            type="primary" if is_current_page else "secondary",
        ):
            st.session_state.page = page_name
            st.rerun()

    page = st.session_state.page

    st.sidebar.divider()
    if st.session_state.user:
        st.sidebar.caption("Account")
        st.sidebar.write(f"Signed in as {st.session_state.user['name']}")
        st.sidebar.write(f"Role: {st.session_state.role.title()}")
        if st.sidebar.button("Logout"):
            logout()
    else:
        st.sidebar.caption("Account")
        st.sidebar.write("Not signed in")
        st.sidebar.info("Log in or register to save crop-yield predictions.")

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
