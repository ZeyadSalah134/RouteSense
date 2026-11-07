# RouteSense

**Predict delays. Protect customer trust.**

RouteSense is an AI-powered delivery risk intelligence
application built with Streamlit and scikit-learn.

## Overview

The application predicts whether an e-commerce order
may arrive after its estimated delivery date.

Dataset: Brazilian E-Commerce Public Dataset by Olist.

## Features

- Delivery risk assessment
- Late-delivery probability estimates
- Operational risk classification
- Model evaluation dashboard
- Feature importance visualization
- Batch prediction with CSV export
- Responsive dark analytics interface

## Technology

- Python
- Streamlit
- Pandas and NumPy
- scikit-learn
- Joblib
- Plotly

## Model

The application loads the original trained Random Forest
pipeline. Model training does not take place in Streamlit.

Expected files in `model_artifacts/`:

- olist_late_delivery_model.joblib
- model_metadata.json
- test_metrics.json
- feature_importance.csv

## Local setup

Create a Python 3.11 environment.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

1. Create a GitHub repository.
2. Upload the application files and model artifacts.
3. Open Streamlit Community Cloud.
4. Connect the GitHub repository.
5. Select `app.py` as the entry point.
6. Deploy using Python 3.11 and the pinned dependencies.

## Limitations

The model uses historical Olist data and payment features.
Its chronological test results show weak predictive
performance. RouteSense is a research prototype and should
not be used for production decisions without further
validation.

## Dataset

https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
