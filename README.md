FYP Phase 1 – Network Intrusion Detection System (NIDS) Prototype

This repository contains the Phase 1 prototype of my Final Year Project.
The purpose of this phase is to demonstrate a working machine-learning-based Network Intrusion Detection System (NIDS) using benchmark network traffic data.

The prototype confirms that the data processing pipeline, trained model, and detection logic are correctly implemented and integrated into an interactive application.

1. Model Training (Development Phase - Author Only)

Model training and data preparation were carried out by the author using Google Colab as part of the development process.

Google Drive was used only during development to store the dataset and generated model artifacts.

The training notebook FYP_Phase1_Colab.ipynb is included for reference and evidence of development progress only.

Assessors are not expected to run this notebook, access Google Drive, or retrain the model.

2. Running the Prototype Application (Demonstration Phase)

The prototype application runs independently of Google Colab and Google Drive.

Local Execution

From the project root directory:

pip install -r requirements.txt
streamlit run app.py
3. Artifacts and Model Loading

The application loads the trained model and preprocessing files from the following local directory:

artifacts/

This directory contains:

rf_phase1_model.pkl – trained Random Forest model

scaler_phase1.pkl – feature scaling parameters

features_phase1.json – selected feature list

meta_phase1.json – configuration and threshold information

demo_traffic_anonymised.csv – example input file 

All required files are bundled with the application.
No external credentials, cloud storage access, or dataset files are required to run the prototype.

4. Prototype Functionality

The prototype allows the user to:

Upload network traffic CSV files (CIC-IDS2017 format)

Detect anomalous network flows using a machine-learning model

View a summary of detected attacks and benign traffic

Inspect the most suspicious network flows

Export detection results as a CSV file

A simple rule-based PortScan heuristic is included alongside the machine-learning anomaly detection for demonstration purposes.

5. Hosting and Access

For assessment purposes, the prototype can be hosted on an external platform (for example, Streamlit Cloud).
The system is fully self-contained and accessible without requiring Google Drive, Google Colab, or access to the original dataset.

Academic Clarification

The dataset is used only during the development and training phase.
For assessment and demonstration, the deployed prototype runs independently using pre-trained model artifacts.

Summary:

This Phase-1 prototype demonstrates:

Successful training of a machine-learning-based NIDS on benchmark data

End-to-end integration of the trained model into a working application

A functional and accessible prototype suitable for the Interim Progression Demonstration (IPD)
