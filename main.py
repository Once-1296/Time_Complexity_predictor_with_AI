import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import sys

def analyze_complexity(csv_file_path):
    """
    Reads a CSV file with 'InputSize' and 'Time', plots the data,
    and predicts the best-fitting time complexity using linear regression.
    """
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{csv_file_path}' is empty.")
        return
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if 'InputSize' not in df.columns or 'Time' not in df.columns:
        print("Error: CSV must have columns named 'InputSize' and 'Time'.")
        return
    
    if df.empty:
        print("Error: The CSV file has no data.")
        return

    n = df['InputSize'].values
    time = df['Time'].values

    features = {
        "O(log n)": np.log2(n),
        "O(sqrt n)": np.sqrt(n),
        "O(n)": n,
        "O(n log n)": n * np.log2(n),
        "O(n sqrt n)": n * np.sqrt(n),
        "O(n^2)": n**2,
        "O(n^3)": n**3
    }

    models = {}
    predictions = {}
    scores = {}

    print("Analyzing data...")

    # baseline constant model (predict mean)
    const_pred = np.full_like(time, np.mean(time), dtype=float)
    const_score = r2_score(time, const_pred)

    for key, feat in features.items():

        feat = feat.reshape(-1, 1)
        # if feature has near-zero variance, skip fitting and mark as poor fit
        if np.allclose(feat.flatten(), feat.flatten()[0]):
            models[key] = None
            predictions[key] = const_pred
            scores[key] = -np.inf
            continue

        try:
            model = LinearRegression()
            model.fit(feat, time)
            pred = model.predict(feat)
            score = r2_score(time, pred)
            models[key] = model
            predictions[key] = pred
            scores[key] = score
        except Exception:
            models[key] = None
            predictions[key] = const_pred
            scores[key] = -np.inf

    best_fit = max(scores, key=scores.get)
    best_score = scores[best_fit]

    print("\n--- Model Fit Analysis (R-squared) ---")
    for key, score in scores.items():
        print(f"{key}: {score:.4f}")
    print("----------------------------------------")

    R_SQUARED_THRESHOLD = 0.8
    if best_score < R_SQUARED_THRESHOLD:
        print(f"\nPredicted Complexity: Non-polynomial or poor fit")
        print(f"(Best model {best_fit} had R^2 = {best_score:.4f}, which is below the {R_SQUARED_THRESHOLD} threshold)")
    else:
        print(f"\nPredicted Time Complexity: {best_fit} (R^2: {best_score:.4f})")

    plt.figure(figsize=(12, 7))
    plt.scatter(n, time, label='Actual Data', alpha=0.6)

    colors = {
        "O(log n)": "purple",
        "O(sqrt n)": "cyan",
        "O(n)": "green",
        "O(n log n)": "orange",
        "O(n sqrt n)": "brown",
        "O(n^2)": "red",
        "O(n^3)": "magenta"
    }

    sort_indices = n.argsort()
    n_sorted = n[sort_indices]

    for key, pred in predictions.items():
        pred_sorted = pred[sort_indices]
        label = f"{key} Fit (R^2={scores.get(key, float('nan')):.4f})"
        plt.plot(n_sorted, pred_sorted, label=label, color=colors.get(key, "black"), linewidth=2)

    plt.xlabel('Input Size (n)')
    plt.ylabel('Execution Time (s)')
    plt.title('Time Complexity Analysis')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_complexity.py <path_to_csv_file>")
        sys.exit(1)
    csv_file = sys.argv[1]
    analyze_complexity(csv_file)
