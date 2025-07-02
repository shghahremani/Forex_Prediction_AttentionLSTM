# 💱 ALFA: Attention-based LSTM for Forex Rate Prediction

This repository provides the implementation for our paper:

**“Prediction of Foreign Currency Exchange Rates Using an Attention-based Long Short-Term Memory Network (ALFA)”**  
📄 _Published in Machine Learning with Applications, 2025_

> **Authors:** Shahram Ghahremani, Uyen Trang Nguyen  
> 📌 York University, Canada  
> 🔗 [DOI Link](https://doi.org/10.1016/j.mlwa.2025.100648)

---

## 📌 Abstract

We introduce **ALFA**, a novel deep learning model combining LSTM with an attention mechanism to predict foreign exchange (Forex) rates. The attention module enables the model to focus on the most relevant temporal patterns, significantly improving predictive accuracy compared to baseline models like GRU, Bi-LSTM, TCN, and Transformer.

---

## 🧪 How to Replicate the Experiments

### 1. 📉 Download Historical Forex Data

You can retrieve historical Forex data using MetaTrader 5:

```bash
python download_historical_data.py
```

This will download EURUSD hourly data (from June 8, 2024 to Sept 8, 2024) and save it to:
```
data/hourly/EURUSD_RECENT.csv
```

👉 Make sure your MetaTrader 5 terminal is installed and configured properly with correct credentials in the script.

---

### 2. 🏋️‍♂️ Train RNN-Based Models

Before training:
- Configure parameters in `common_variables.py` (e.g., `interval`, `Problem`, `seed`, `batch_size`, etc.)

To start training, run:

```bash
python Training_Regression.py
```

The training script supports various architectures:
- `LSTM`
- `GRU`
- `Bidirectional`
- `Attention` (ALFA)
- `stack_lstm`
- `TCN`
- `Transformer`

Trained models are saved in:
```
models/{ticker}/{interval}/{Problem}/{model_name}/{feature}-{window_size}.h5
```

---

### 3. 📊 Evaluate Trained Models

Run the evaluation script to benchmark models and generate performance metrics:

```bash
python Evalution_hourly_for_paper.py
```

This will:
- Evaluate MAE, RMSE, MAPE
- Compute confidence intervals
- Log inference times
- Output results as CSV files under:
  ```
  results/{ticker}/{model_name}/{feature}_best_results.csv
  results/{ticker}/{model_name}/all_results.csv
  ```

---

## 📁 Project Structure

```bash
.
├── download_historical_data.py      # Downloads Forex data via MetaTrader5
├── Training_Regression.py           # Trains RNN models
├── Evalution_hourly_for_paper.py    # Evaluates trained models
├── common_variables.py              # Global config (seed, batch size, paths)
├── models/                          # Saved model checkpoints
├── results/                         # Evaluation results
└── data/                            # Historical Forex data
```

---

## 📈 Sample Results

ALFA achieved:
- **MAE: 0.887 (EUR/USD)** with {Low, Close}
- **MAE: 0.176 (USD/JPY)** with {High, Open}
- **MAE: 0.177 (GBP/JPY)** with all 4 features

📌 Average Inference Time: **~0.55 ms/sample**  
📊 Estimated Annual Return: **8.79%–19%**, comparable to professional traders.

---

## 🔧 Dependencies

- `TensorFlow`, `Keras`
- `scikit-learn`, `pandas`, `numpy`
- `ta` (technical analysis library)
- `MetaTrader5` Python binding

Install with:

```bash
pip install -r requirements.txt
```

---

## 📜 License

MIT License.  
© 2025 Shahram Ghahremani & Uyen Trang Nguyen
