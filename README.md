# ML-Retail-Forecasting-Clustering
"A dual-domain machine learning capstone: predicting Walmart store sales using Random Forest and segmenting online retail customers via RFM and K-Means clustering."


# Dual-Domain Retail Analytics: Sales Forecasting & Behavioral Customer Segmentation

Welcome to my first comprehensive Data Science Capstone Project portfolio! This repository showcases an end-to-end machine learning solution addressing two critical challenges in modern retail operations: supply chain demand-forecasting and data-driven customer relationship management (CRM).

## 🚀 Project Overview

This project is divided into two distinct machine learning tracks:

*   **Part A: Walmart Store Sales Analysis & 12-Week Forecasting** 
    *   **Objective:** Solve retail inventory mismatches by predicting future weekly demand.
    *   **Methodology:** Explored historical sales sequences across multiple stores alongside holiday waves and macroeconomic factors (CPI, Unemployment, Fuel Prices). Engineered explicit temporal features (`Year`, `Month`, `WeekOfYear`) to feed a robust **Random Forest Regressor** ensemble model.
*   **Part B: Online Retail Customer Analysis & Behavioral Segmentation**
    *   **Objective:** Uncover hidden purchasing patterns to drive personalized marketing and retention.
    *   **Methodology:** Handled transaction anomalies, engineered **Recency, Frequency, and Monetary (RFM)** behavioral metrics, scaled the vectors using log-transformation and standard normalization, and implemented an unsupervised **K-Means Clustering** algorithm.

---

## 📂 Repository Structure

```text
├── data/
│   └── walmart_12_week_forecast.csv           # Final 12-week predictive outputs
├── models/
│   ├── sales_forecasting_rf_model.pkl         # Saved Random Forest Regressor
│   └── customer_segmentation_kmeans_model.pkl # Saved K-Means Clustering model
├── notebooks/
│   └── Capstone_Project.ipynb                  # Complete interactive Google Colab notebook
├── reports/
│   └── Capstone_Project_Submission_Report.pdf # Formal, academic-style compilation report
├── README.md                                  # Repository documentation
└── requirements.txt                           # Python library dependencies
