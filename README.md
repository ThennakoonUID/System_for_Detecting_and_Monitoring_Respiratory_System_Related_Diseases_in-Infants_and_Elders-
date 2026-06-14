# VitaSync: Respiratory Disease Detection & Monitoring System

## Overview
**VitaSync** is an affordable, wearable monitoring system designed to detect and manage Chronic Respiratory Diseases (CRDs) such as Asthma, Chronic Obstructive Pulmonary Disease (COPD), and Sleep Apnea in infants and elders. Developed for the Engineering Design Project (BM1190) at the University of Moratuwa, this system shifts healthcare from reactive treatment to proactive management by providing continuous, non-invasive point-of-care data logging. 

## The Problem & Impact
Chronic Respiratory Diseases account for 12% of mortality in South Asia. Currently, early symptoms are frequently neglected in rural and low-income populations due to the centralization of high-cost diagnostic tools like hospital-grade spirometers. VitaSync addresses this critical diagnostic gap by offering a highly accessible, low-cost alternative that empowers patients and doctors to monitor conditions early and reliably.

## Key Features
* **Multiparameter Monitoring:** Continuously tracks $SpO_{2}$, heart rate, body temperature, and respiratory rate.
* **Physical Effort Tracking:** Utilizes an accelerometer to measure the physical expansion and contraction of the chest or neck to accurately log breathing rates.
* **Intelligent Analysis:** Implements a Machine Learning model to analyze raw sensor data, identify dangerous drops in oxygen levels, and detect early disease trends.
* **Mobile Dashboard:** Pairs with a mobile application to provide real-time visualization of "Respiration Status," "Oxygen Level," and "Breathing Rate" histories.
* **Emergency Alert System:** Triggers automated SMS alerts alongside real-time visual and audio warnings for necessary medical personnel or caretakers during critical events.

## Hardware Architecture
* **Microcontroller:** Low-cost ESP32 or Arduino-based processing unit.
* **Sensors:** High-sensitivity pulse oximetry module, heart-rate module, temperature sensor, and accelerometer.
* **Power Supply:** Rechargeable Li-Po battery system.
* **Form Factor:** Compact, battery-powered wearable wristband and chest strap ensuring patient comfort and ease of use.

## Project Team
**Team VitaSync** — Department of Electronic and Telecommunication Engineering (Biomedical Engineering), University of Moratuwa.
* Thennakoon U.I.D. (240657J)
* Pitiduwa S.M. (240518K)
* Tennekone A.D.T.M.S.H. (240642J)
* Wanninayaka W.M.I.R. (240695X)
