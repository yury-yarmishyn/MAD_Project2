# 🎶 Track Genre Analysis with Machine Learning

A machine learning project that classifies music tracks into genres based on various audio features.

## 📌 Project Overview

The goal of this project is to accurately classify music tracks by genre using machine learning models trained on audio features such as tempo, energy, and danceability.

## 💻 Technologies

- **Python**
- **Scikit-Learn**
- **Numpy**
- **Pandas**

## 📊 Machine Learning Models

- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**
- **Random Forest**

## 🔍 Features and Workflow

### 1. Dataset
This project uses a dataset containing Spotify track data with detailed audio features for each track, allowing for genre classification.

### 2. Data Processing
The dataset undergoes pre-processing, including cleaning and feature engineering, to prepare the data for model input.

### 3. Model Training and Evaluation
Multiple machine learning models are trained to classify tracks, with each model’s performance being evaluated based on predictive accuracy. The best-performing model is selected for final predictions.

### 4. Genre Prediction
Given the audio features, the chosen model predicts the genre and provides a confidence score.

## 🚀 How to Use

1. **Visit Musicstax**: Go to [musicstax.com](https://musicstax.com) and find a track you're interested in.
2. **Input Track Data**: Enter the track's data into the `new_track` variable.
3. **Run the Script**: Start the script to get predictions (Note: The initial run may take longer due to model training. For faster results, you can comment out the training for `RandomForest` and `Hybrid` models).

---

🎉 **Enjoy classifying your favorite tracks with this project!** 🎉
