from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT

BG = "#0B1220"
SURFACE = "#152234"
SURFACE2 = "#1E3046"
ACCENT = "#69B5D1"
TEXT = "#F4F8FC"
MUTED = "#9FB2C5"

st.set_page_config(
    page_title="RouteSense | Delivery Risk Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

html, body, [data-testid="stApp"] {
    font-family: 'DM Sans', sans-serif;
    color: #F4F8FC;
}
[data-testid="stApp"], [data-testid="stHeader"] {
    background: #0B1220;
}
[data-testid="stSidebar"] {
    background: #111D2C;
    border-right: 1px solid #2A3D52;
}
h1, h2, h3, h4 {
    font-family: 'Manrope', sans-serif;
    letter-spacing: -0.03em;
}
.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}
[data-testid="stMetric"] {
    background: #152234;
    border: 1px solid #2A3D52;
    padding: 18px;
    border-radius: 12px;
}
[data-testid="stMetricLabel"] {
    color: #9FB2C5;
}
[data-testid="stMetricValue"] {
    font-family: 'Manrope', sans-serif;
    color: #F4F8FC;
}
div.stButton > button[kind="primary"],
div[data-testid="stFormSubmitButton"] button {
    background: #69B5D1;
    color: #0B1220;
    border: none;
    border-radius: 9px;
    font-weight: 700;
}
div.stButton > button[kind="primary"]:hover,
div[data-testid="stFormSubmitButton"] button:hover {
    background: #8CCCE2;
    color: #0B1220;
}
.rs-panel {
    background: #152234;
    border: 1px solid #2A3D52;
    border-radius: 14px;
    padding: 24px;
    margin: 15px 0;
}
.rs-eyebrow {
    color: #69B5D1;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}
.rs-muted {
    color: #9FB2C5;
    line-height: 1.7;
}
.rs-risk {
    background: #152234;
    border: 1px solid #31506A;
    border-radius: 16px;
    padding: 28px;
    margin-top: 20px;
}
.rs-number {
    font: 800 3.2rem 'Manrope', sans-serif;
    letter-spacing: -0.06em;
}
.rs-chip {
    display: inline-block;
    border: 1px solid #345D72;
    color: #9DDAF1;
    background: #1B3548;
    border-radius: 99px;
    padding: 5px 11px;
    font-size: .78rem;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load(
        ARTIFACTS / "olist_late_delivery_model.joblib"
    )


@st.cache_data
def load_json(filename):
    with open(ARTIFACTS / filename, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_importance():
    return pd.read_csv(
        ARTIFACTS / "feature_importance.csv"
    )


required_files = [
    "olist_late_delivery_model.joblib",
    "model_metadata.json",
    "test_metrics.json",
    "feature_importance.csv",
]

missing_files = [
    name for name in required_files
    if not (ARTIFACTS / name).exists()
]

if missing_files:
    st.error("Model files are missing: " + ", ".join(missing_files))
    st.stop()

try:
    model = load_model()
    metadata = load_json("model_metadata.json")
    metrics = load_json("test_metrics.json")
    importance = load_importance()
except Exception as exc:
    st.error("Could not load the exported model files.")
    st.exception(exc)
    st.stop()

FEATURES = metadata["features"]
NUMERIC = metadata["numeric_features"]
CATEGORICAL = metadata["categorical_features"]
MODEL_THRESHOLD = float(metadata["classification_threshold"])

if list(model.feature_names_in_) != FEATURES:
    st.error("Model and metadata feature order do not match.")
    st.stop()

if 1 not in model.classes_:
    st.error("The loaded classifier does not contain class 1.")
    st.stop()

POS_INDEX = list(model.classes_).index(1)


def classify(probability, medium, high):
    if probability >= high:
        return "High"
    if probability >= medium:
        return "Medium"
    return "Low"


def action_for_risk(risk):
    actions = {
        "High": (
            "Prioritize operational review and consider "
            "proactive customer communication."
        ),
        "Medium": (
            "Monitor fulfillment progress and review "
            "the order if new concerns appear."
        ),
        "Low": "Continue normal fulfillment monitoring."
    }
    return actions[risk]


def validate_data(data):
    missing = [c for c in FEATURES if c not in data.columns]
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    result = data.loc[:, FEATURES].copy()

    for col in NUMERIC:
        raw = result[col]
        result[col] = pd.to_numeric(raw, errors="coerce")
        if result[col].isna().any():
            raise ValueError(
                f"Invalid or missing numeric values in '{col}'."
            )
        if not np.isfinite(result[col].astype(float)).all():
            raise ValueError(f"Non-finite values in '{col}'.")

    for col in CATEGORICAL:
        if result[col].isna().any():
            raise ValueError(f"Missing values in '{col}'.")
        result[col] = result[col].astype(str).str.strip()
        if result[col].eq("").any():
            raise ValueError(f"Empty values in '{col}'.")

    ranges = {
        "item_count": (1, None),
        "unique_sellers": (1, None),
        "total_price": (0, None),
        "total_freight": (0, None),
        "mean_item_price": (0, None),
        "mean_freight": (0, None),
        "payment_total": (0, None),
        "payment_installments_max": (0, None),
        "payment_methods": (0, None),
        "estimated_lead_days": (0, None),
        "purchase_month": (1, 12),
        "purchase_dayofweek": (0, 6),
        "purchase_hour": (0, 23),
        "purchase_is_weekend": (0, 1),
    }

    for col, (lower, upper) in ranges.items():
        if col not in result:
            continue
        if (result[col] < lower).any():
            raise ValueError(f"'{col}' must be at least {lower}.")
        if upper is not None and (result[col] > upper).any():
            raise ValueError(f"'{col}' must not exceed {upper}.")

    integer_cols = [
        "item_count", "unique_sellers", "purchase_year",
        "purchase_month", "purchase_dayofweek", "purchase_hour",
        "purchase_is_weekend", "payment_installments_max",
        "payment_methods"
    ]
    for col in integer_cols:
        if col in result and not (result[col] % 1 == 0).all():
            raise ValueError(f"'{col}' must contain whole numbers.")

    if "purchase_dayofweek" in result and "purchase_is_weekend" in result:
        expected = result["purchase_dayofweek"].isin([5, 6]).astype(int)
        if not (expected == result["purchase_is_weekend"]).all():
            raise ValueError(
                "Weekend indicator does not match purchase day of week."
            )

    if "item_count" in result and "mean_item_price" in result:
        expected = result["total_price"] / result["item_count"]
        if not np.allclose(
            expected, result["mean_item_price"], atol=0.02
        ):
            raise ValueError(
                "mean_item_price does not match total_price / item_count."
            )

    if "item_count" in result and "mean_freight" in result:
        expected = result["total_freight"] / result["item_count"]
        if not np.allclose(
            expected, result["mean_freight"], atol=0.02
        ):
            raise ValueError(
                "mean_freight does not match total_freight / item_count."
            )

    return result


def predict(data):
    clean = validate_data(data)
    return model.predict_proba(clean)[:, POS_INDEX]


def heading(eyebrow, title, subtitle):
    st.markdown(
        f'<div class="rs-eyebrow">{eyebrow}</div>',
        unsafe_allow_html=True
    )
    st.title(title)
    st.markdown(
        f'<p class="rs-muted">{subtitle}</p>',
        unsafe_allow_html=True
    )


def show_metric(title, value, explanation):
    st.metric(title, value)
    st.caption(explanation)


with st.sidebar:
    st.markdown("## ◈ RouteSense")
    st.caption("DELIVERY RISK INTELLIGENCE")
    st.divider()

    page = st.radio(
        "Workspace",
        [
            "Overview",
            "Risk Assessment",
            "Model Insights",
            "Batch Prediction",
            "About RouteSense"
        ],
        label_visibility="collapsed"
    )

    st.divider()
    st.caption("MODEL STATUS")
    st.markdown("**Random Forest · Loaded**")
    st.caption(f"{len(FEATURES)} input features")
    st.caption("Historical Olist research prototype")

    st.divider()
    st.caption("OPERATIONAL RISK BANDS")

    medium_risk = st.slider(
        "Medium risk starts at",
        min_value=0.05,
        max_value=0.85,
        value=0.30,
        step=0.05
    )

    high_risk = st.slider(
        "High risk starts at",
        min_value=medium_risk + 0.05,
        max_value=0.99,
        value=max(0.60, round(medium_risk + 0.05, 2)),
        step=0.01
    )

    st.caption(
        "These are business triage bands, not the model's "
        "validated classification threshold."
    )


if page == "Overview":
    heading(
        "DELIVERY INTELLIGENCE",
        "Predict delays. Protect customer trust.",
        "RouteSense turns historical e-commerce order "
        "information into delivery risk estimates."
    )

    st.markdown(
        '<span class="rs-chip">Research prototype · Olist Brazil</span>',
        unsafe_allow_html=True
    )

    b, c, d = st.columns(3)

    with b:
        show_metric(
            "Test Recall",
            f'{metrics["test_recall"]:.1%}',
            "Late deliveries identified"
        )
    with c:
        show_metric(
            "Test ROC-AUC",
            f'{metrics["test_roc_auc"]:.3f}',
            "Ranking discrimination"
        )
    with d:
        show_metric(
            "Test PR-AUC",
            f'{metrics["test_pr_auc"]:.3f}',
            "Precision-recall performance"
        )

    st.warning(
        "The held-out test performance is weak. "
        "RouteSense is a research demonstration, not "
        "a validated production decision system."
    )

    left, right = st.columns([1.1, 1])

    with left:
        st.markdown("""
        <div class="rs-panel">
        <h3>What RouteSense does</h3>
        <p>RouteSense estimates the probability that an
        e-commerce order will arrive after its estimated
        delivery date. Teams can use the output to prioritize
        orders for manual review.</p>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class="rs-panel">
        <h3>Machine learning approach</h3>
        <p>Random Forest classifier with preprocessing,
        hyperparameter tuning, chronological data splitting,
        and validation-based threshold selection.</p>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("How it works")
    st.markdown(
        "**1. Enter order details** → "
        "**2. Validate the features** → "
        "**3. Calculate delivery risk** → "
        "**4. Recommend a review action**"
    )

    st.info(
        "The model uses historical Olist data. "
        "It does not provide live shipment tracking."
    )


elif page == "Risk Assessment":
    heading(
        "ORDER INTELLIGENCE",
        "Delivery Risk Assessment",
        "Enter the information available at checkout "
        "to estimate the probability of late delivery."
    )

    defaults = metadata.get("numeric_defaults", {})
    cat_defaults = metadata.get("categorical_defaults", {})
    options = metadata.get("categorical_options", {})

    with st.form("risk_form"):
        st.subheader("Order details")

        col1, col2, col3 = st.columns(3)

        with col1:
            item_count = st.number_input(
                "Number of items", min_value=1, value=1, step=1
            )
            total_price = st.number_input(
                "Total item price (BRL)",
                min_value=0.0,
                value=float(defaults.get("total_price") or 120),
                step=10.0
            )
            total_freight = st.number_input(
                "Total freight (BRL)",
                min_value=0.0,
                value=float(defaults.get("total_freight") or 20),
                step=1.0
            )

        with col2:
            unique_sellers = st.number_input(
                "Number of sellers",
                min_value=1, max_value=int(item_count),
                value=1, step=1
            )
            estimated_lead_days = st.number_input(
                "Estimated delivery lead time (days)",
                min_value=0.0,
                value=float(
                    defaults.get("estimated_lead_days") or 20
                ),
                step=1.0
            )

        with col3:
            customer_state = st.selectbox(
                "Customer state",
                options.get("customer_state", ["SP"]),
                index=0
            )
            main_category = st.selectbox(
                "Main product category",
                options.get("main_category", ["other"]),
                index=0
            )
            main_seller_state = st.selectbox(
                "Main seller state",
                options.get("main_seller_state", ["SP"]),
                index=0
            )

        st.divider()
        st.subheader("Purchase timing")

        purchase_date = st.date_input(
            "Purchase date",
            value=pd.Timestamp("2018-05-15").date(),
            help="Use a historical example consistent with "
                 "the training period for your demonstration."
        )

        purchase_hour = st.slider(
            "Purchase hour", 0, 23, 14
        )

        st.divider()
        st.subheader("Payment information")

        pay1, pay2, pay3 = st.columns(3)

        with pay1:
            payment_total = st.number_input(
                "Payment total (BRL)",
                min_value=0.0,
                value=float(defaults.get("payment_total") or 150),
                step=10.0
            )
        with pay2:
            installments = st.number_input(
                "Maximum installments",
                min_value=0,
                value=int(
                    defaults.get("payment_installments_max") or 1
                ),
                step=1
            )
        with pay3:
            payment_methods = st.number_input(
                "Number of payment methods",
                min_value=0,
                value=int(defaults.get("payment_methods") or 1),
                step=1
            )

        st.caption(
            "Average item price and average freight are "
            "calculated automatically to keep the input "
            "features consistent."
        )

        submitted = st.form_submit_button(
            "Assess Delivery Risk",
            use_container_width=True
        )

    if submitted:
        order = {
            "item_count": int(item_count),
            "total_price": float(total_price),
            "total_freight": float(total_freight),
            "mean_item_price": float(total_price / item_count),
            "mean_freight": float(total_freight / item_count),
            "unique_sellers": int(unique_sellers),
            "purchase_year": int(purchase_date.year),
            "purchase_month": int(purchase_date.month),
            "purchase_dayofweek": int(purchase_date.weekday()),
            "purchase_hour": int(purchase_hour),
            "purchase_is_weekend": int(
                purchase_date.weekday() >= 5
            ),
            "estimated_lead_days": float(estimated_lead_days),
            "payment_total": float(payment_total),
            "payment_installments_max": int(installments),
            "payment_methods": int(payment_methods),
            "customer_state": customer_state,
            "main_category": main_category,
            "main_seller_state": main_seller_state,
        }

        try:
            frame = pd.DataFrame([order])[FEATURES]
            probability = float(predict(frame)[0])
            risk = classify(
                probability, medium_risk, high_risk
            )

            colors = {
                "Low": "#70C9A5",
                "Medium": "#E8C779",
                "High": "#F08D8D"
            }

            st.markdown(
                f"""
                <div class="rs-risk">
                    <div class="rs-eyebrow">PREDICTION RESULT</div>
                    <div class="rs-number"
                         style="color:{colors[risk]}">
                        {probability:.1%}
                    </div>
                    <h3>{risk} delivery risk</h3>
                    <p class="rs-muted">
                        Estimated probability of arriving
                        later than the expected delivery date.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(probability)

            st.markdown("#### Recommended next action")
            st.info(action_for_risk(risk))

            classification = int(
                probability >= MODEL_THRESHOLD
            )

            st.caption(
                f"Model classification threshold: "
                f"{MODEL_THRESHOLD:.3f}. "
                f"Binary prediction: {classification} "
                f"(1 = late, 0 = on-time)."
            )

            st.caption(
                "This is an estimate, not a guarantee. "
                "Risk bands guide operational triage and "
                "do not prove that an order will be late."
            )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


elif page == "Model Insights":
    heading(
        "MODEL TRANSPARENCY",
        "Model Insights",
        "Evaluation results from the held-out "
        "chronological test set."
    )

    names = {
        "Accuracy": "test_accuracy",
        "Precision": "test_precision",
        "Recall": "test_recall",
        "F1-score": "test_f1",
        "ROC-AUC": "test_roc_auc",
        "PR-AUC": "test_pr_auc"
    }

    metric_data = pd.DataFrame([
        {"Metric": name, "Score": metrics[key]}
        for name, key in names.items()
        if key in metrics
    ])

    fig = px.bar(
        metric_data,
        x="Metric",
        y="Score",
        text="Score",
        color_discrete_sequence=[ACCENT]
    )
    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside"
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        yaxis_range=[0, 1.05],
        height=380
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Accuracy measures overall correctness. "
        "Precision measures how often predicted late "
        "orders are truly late. Recall measures how "
        "many actual late orders are detected. "
        "F1 balances precision and recall. "
        "ROC-AUC measures ranking across thresholds; "
        "PR-AUC summarizes precision-recall performance."
    )

    st.subheader("Confusion Matrix")

    matrix = metrics.get("confusion_matrix")

    if matrix:
        fig_cm = go.Figure(
            go.Heatmap(
                z=matrix,
                x=["Predicted on-time", "Predicted late"],
                y=["Actual on-time", "Actual late"],
                text=matrix,
                texttemplate="%{text}",
                colorscale=[
                    [0, SURFACE],
                    [1, ACCENT]
                ],
                showscale=False
            )
        )
        fig_cm.update_layout(
            template="plotly_dark",
            paper_bgcolor=BG,
            height=350
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    else:
        st.info("Confusion matrix unavailable.")

    st.subheader("Feature Importance")

    numeric_columns = importance.select_dtypes(
        include="number"
    ).columns.tolist()

    importance_col = next(
        (c for c in numeric_columns if "mean" in c.lower()),
        numeric_columns[0] if numeric_columns else None
    )

    feature_col = next(
        (
            c for c in importance.columns
            if c not in numeric_columns
        ),
        None
    )

    if importance_col and feature_col:
        top_features = (
            importance.nlargest(15, importance_col)
            .sort_values(importance_col)
        )
        fig_imp = px.bar(
            top_features,
            x=importance_col,
            y=feature_col,
            orientation="h",
            color_discrete_sequence=[ACCENT]
        )
        fig_imp.update_layout(
            template="plotly_dark",
            paper_bgcolor=BG,
            plot_bgcolor=BG,
            height=520
        )
        st.plotly_chart(
            fig_imp, use_container_width=True
        )
    else:
        st.dataframe(importance, use_container_width=True)

    st.caption(
        "Permutation importance measures changes in "
        "model performance when features are shuffled. "
        "It is global model information, not a causal "
        "explanation of an individual prediction."
    )

    with st.expander("Model details"):
        st.json({
            "model": metadata.get("model_type"),
            "features": len(FEATURES),
            "test_samples": metrics.get("test_samples"),
            "classification_threshold": MODEL_THRESHOLD,
            "training_start": metadata.get("training_start_date"),
            "training_end": metadata.get("training_end_date"),
            "best_parameters": metadata.get("best_parameters"),
        })


elif page == "Batch Prediction":
    heading(
        "BULK OPERATIONS",
        "Batch Prediction",
        "Upload a CSV of orders to evaluate several "
        "orders with the same trained pipeline."
    )

    template = pd.DataFrame(columns=FEATURES)

    st.download_button(
        "Download CSV Template",
        data=template.to_csv(index=False),
        file_name="routesense_input_template.csv",
        mime="text/csv"
    )

    uploaded = st.file_uploader(
        "Upload order CSV",
        type=["csv"]
    )

    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)

            if raw.empty:
                raise ValueError("The uploaded CSV is empty.")

            st.caption(f"Rows uploaded: {len(raw):,}")
            st.dataframe(raw.head(10), use_container_width=True)

            missing = [
                col for col in FEATURES
                if col not in raw.columns
            ]

            if missing:
                st.error(
                    "Missing required features: "
                    + ", ".join(missing)
                )
            elif st.button(
                "Run Batch Prediction",
                type="primary"
            ):
                probabilities = predict(raw)

                output = raw.copy()
                output["late_probability"] = probabilities
                output["risk_level"] = [
                    classify(p, medium_risk, high_risk)
                    for p in probabilities
                ]
                output["predicted_late"] = (
                    probabilities >= MODEL_THRESHOLD
                ).astype(int)
                output["recommended_action"] = [
                    action_for_risk(level)
                    for level in output["risk_level"]
                ]

                st.success(
                    f"Predictions completed for "
                    f"{len(output):,} orders."
                )

                st.dataframe(
                    output.head(100),
                    use_container_width=True
                )

                st.download_button(
                    "Download Prediction Results",
                    data=output.to_csv(index=False),
                    file_name="routesense_predictions.csv",
                    mime="text/csv"
                )

        except Exception as exc:
            st.error(str(exc))

    st.caption(
        "All required features must be supplied. "
        "The application will not fill in missing "
        "order information or fabricate predictions."
    )


elif page == "About RouteSense":
    heading(
        "ABOUT THE PROJECT",
        "About RouteSense",
        "A machine learning approach to identifying "
        "delivery risk in e-commerce."
    )

    st.markdown("""
    ### The business problem

    Late deliveries can increase customer complaints,
    disrupt customer expectations and create additional
    work for fulfillment teams.

    RouteSense explores whether order information
    available around checkout can help identify
    potentially delayed deliveries.

    ### Dataset

    The project uses the Brazilian E-Commerce Public
    Dataset by Olist, containing historical transactions
    from Brazilian e-commerce marketplaces.

    ### Machine learning approach

    The target is binary:

    - **0:** On-time delivery
    - **1:** Late delivery

    The model is a Random Forest classifier trained
    using preprocessing, feature engineering,
    hyperparameter tuning and chronological
    train/validation/test splitting.

    The original preprocessing pipeline is preserved
    inside the exported model.

    ### Business applications

    RouteSense can support exploratory risk
    prioritization, operational review and research
    into earlier identification of delivery issues.

    The suggested actions do not automatically make
    decisions or contact customers.

    ### Limitations

    The dataset is historical and geographically
    specific to Brazil. The model uses payment
    information and requires it to be available
    before a prediction can be made.

    The chronological test results show weak
    predictive performance. The current version
    should not be used for consequential production
    decisions without further improvement and
    validation.

    This application does not use live carrier
    tracking or real-time delivery events.
    """)

    st.markdown(
        "[Olist dataset on Kaggle]"
        "(https://www.kaggle.com/datasets/"
        "olistbr/brazilian-ecommerce)"
    )
