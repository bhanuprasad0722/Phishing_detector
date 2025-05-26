# 🛡️ Phikita - Phishing Website Detection using ABS-CNN

A full-stack Django-based web application that detects phishing websites using a deep learning model (ABS-CNN). This project is powered by a custom-trained neural network and provides real-time prediction, visualization, and user tracking.

---

## 🔍 Project Overview

Phikita helps detect whether a given URL is **legitimate** or a **phishing attempt** using advanced deep learning techniques. It features:

- User Authentication System (User & Admin)
- Real-time URL prediction with confidence score
- Admin dashboard with model accuracy charts and user activity logs
- Dataset upload & retraining interface
- Contact form for user queries

---

## 🧠 Model: ABS-CNN (Attention-Based Convolutional Neural Network)

We developed a custom CNN architecture enhanced with **attention mechanisms** to give higher importance to phishing-relevant features.

### Model Highlights:
- Dual Conv1D layers with BatchNormalization
- Attention mechanism for sequence feature weighting
- Achieved **96.7% accuracy** on test data
- Balanced class weights to handle imbalanced data

---

## 📊 Admin Dashboard Features

- Model Accuracy Comparison Chart
- Prediction Trends Over Time
- Latest Uploaded Dataset Viewer
- Contacted User Queries Section
- Login & Logout Activity Tracker
- Registered Users Management

---

## 📁 Dataset Source

This project uses the dataset from the research paper:

**“Phikita: Phishing Kit Attacks Dataset for Phishing Website Identification”**  
➡️ [Download Paper](https://ieeexplore.ieee.org/document/10103863)

---

## 🧰 Technologies Used

| Stack | Tools |
|-------|-------|
| Backend | Django, Python |
| ML Model | TensorFlow, Keras, Scikit-Learn |
| Frontend | HTML5, CSS3, Chart.js, Matplotlib |
| Deployment | GitHub |
| Database | SQLite3 |

---

## 🚀 How to Run the Project

```bash
# Clone the repository
git clone https://github.com/your-username/phishing-detector.git
cd phishing-detector

# Install dependencies
pip install -r requirements.txt

# Run the Django server
python manage.py runserver
