# Respiratory Risk ML Model

A machine learning model that predicts whether a person is at risk of a respiratory disease by analysing SpO2 (blood oxygen saturation) and respiratory rate readings from a wearable device.

## Project overview

This model is part of the VitaSync wearable health monitoring system, built around an ESP32-WROOM-32 with a MAX30105 pulse oximeter and dual BMI160 IMUs.

The model is trained on the [BIDMC Respiratory Dataset](https://physionet.org/content/bidmc/1.0.0/) (53 ICU patients), using summary statistics extracted from 5-minute windows of SpO2 and RR readings.

## Model performance

- **Algorithm:** Logistic Regression
- **Cross-validation accuracy:** 95.3% ± 5.8%
- **Cross-validation recall:** 93.3% ± 13.3%
- **Cross-validation AUC-ROC:** 97.3% ± 5.3%
- **Training patients:** 42 | **Test patients:** 11

## Risk label definition

A patient is flagged **at risk** if any of the following are true over a 5-minute window:
- Mean SpO2 < 95%
- Minimum SpO2 < 92%
- Mean respiratory rate > 20 breaths/min
- More than 20% of RR readings outside the normal 12–20 bpm range

## Setup

1. Clone the repository
2. Create a virtual environment: `python3 -m venv venv && source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Run notebooks in order to reproduce the model

## Dataset

BIDMC Waveform and Physiologic Signal Collection (PhysioNet).
Raw dataset files are not included — run `01_download.ipynb` to download them.

## Disclaimer

Research prototype only. Not a certified medical device.