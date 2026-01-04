# US Eastern Energy Load Forecasting (PJM) ⚡

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

## 📌 Project Overview
This project focuses on **Time Series Forecasting** to predict hourly energy consumption using the **PJM Interconnection (PJME)** dataset. PJM is a regional transmission organization in the United States that coordinates the movement of wholesale electricity in the Eastern US.

The core objective of this repository is not just to build a prediction model, but to conduct a **comparative analysis on model convergence**, specifically investigating the effects of prolonged training (Overfitting vs. Generalization).

## 📊 The Dataset
* **Source:** PJM Hourly Energy Consumption Data (PJME).
* **Region:** Eastern United States Interconnection.
* **Data Type:** Univariate Time Series (Hourly timestamps).
* **Preprocessing:** Data normalization using `MinMaxScaler` to aid LSTM convergence.

<img width="1732" height="874" alt="150 Epoch (1)" src="https://github.com/user-attachments/assets/078a8fa3-8130-4e4f-a90e-65cd739b4035" />
🧪 The Experiment: 150 vs. 500 Epochs
As a Junior Machine Learning Engineer, a common misconception is that "more training equals better performance." I tested this hypothesis by monitoring the model's behavior over extended epochs.

### Key Observations:
1.  **Diminishing Returns:** After approximately **150 epochs**, the Validation Loss stabilized, while Training Time continued to increase linearly.
2.  **Overfitting Signs:** Approaching **500 epochs**, the Training Loss continued to decrease near-perfectly, but the Validation Loss began to plateau or diverge. This indicates the model started memorizing noise rather than learning patterns.

**Visual Evidence:**
*(The graph below demonstrates the divergence between Training Loss and Validation Loss)*

![Loss Graph - Training vs Validation 500 Epochs]
<img width="1072" height="857" alt="LSTM_500 EPOCH_Full screen" src="https://github.com/user-attachments/assets/1845313a-5cab-40d0-a227-10cbfe651a27" />

🛠️ Tech Stack & Methodology
* **Model Architecture:** LSTM (Long Short-Term Memory) Neural Network.
* **Frameworks:** TensorFlow (Keras), Scikit-Learn.
* **Data Handling:** Pandas, NumPy.
* **Visualization:** Matplotlib.

🚀 How to Run This Project

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/PJME-Energy-Load-Forecasting.git](https://github.com/yourusername/PJME-Energy-Load-Forecasting.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the script:**
    ```bash
    python main_forecasting.py
    ```

📈 Results
The final model (optimized with Early Stopping) successfully captures the cyclical nature of energy demand, accounting for daily peak hours and seasonal trends.

![Prediction Graph](path/to/your/prediction_image.png)

🤝 Feedback
I am actively learning and improving my understanding of Deep Learning dynamics. If you have suggestions regarding the LSTM architecture or Hyperparameter Tuning, please feel free to open an issue or reach out!
