import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import sys

def analyze_complexity(csv_file_path):
    """
    Reads a CSV file with 'InputSize' and 'Time', plots the data,
    and predicts the polynomial time complexity.
    
    Note: The prompt mentioned 'logistic regression', but that is a
    classification algorithm. For fitting a curve to continuous data (time),
    we use 'Linear Regression'. We can fit models like y = a*x, y = a*(x^2), etc.,
    which are all forms of linear regression (linear in the coefficient 'a').
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

    # Prepare data
    # Reshape(-1, 1) is required by scikit-learn
    n = df['InputSize'].values.reshape(-1, 1)
    time = df['Time'].values

    # Create polynomial features
    features = {
        'O(n)': n,
        'O(n^2)': n**2,
        'O(n^3)': n**3
    }

    models = {}
    predictions = {}
    scores = {}

    print("Analyzing data...")

    # Fit a separate model for each complexity
    for key, feat in features.items():
        # Fit a LinearRegression model (e.g., time = a * n^2)
        model = LinearRegression()
        model.fit(feat, time)
        
        # Get predictions and R-squared score
        pred = model.predict(feat)
        score = r2_score(time, pred)
        
        models[key] = model
        predictions[key] = pred
        scores[key] = score

    # Find the best fit based on the highest R-squared value
    best_fit = max(scores, key=scores.get)
    best_score = scores[best_fit]

    print("\n--- Model Fit Analysis (R-squared) ---")
    for key, score in scores.items():
        print(f"{key}: {score:.4f}")
    
    print("----------------------------------------")
    
    # Define a threshold for what we consider a "good fit"
    R_SQUARED_THRESHOLD = 0.8
    
    if best_score < R_SQUARED_THRESHOLD:
        print(f"\nPredicted Complexity: Non-polynomial or poor fit")
        print(f"(Best model {best_fit} had R^2 = {best_score:.4f}, which is below the {R_SQUARED_THRESHOLD} threshold)")
    else:
        print(f"\nPredicted Time Complexity: {best_fit} (R^2: {best_score:.4f})")


    # Plotting
    plt.figure(figsize=(12, 7))
    plt.scatter(n, time, label='Actual Data', alpha=0.6)
    
    # Plot all regression lines for comparison
    colors = {'O(n)': 'green', 'O(n^2)': 'orange', 'O(n^3)': 'red'}
    
    # Sort data for clean plotting of lines
    sort_indices = n.flatten().argsort()
    n_sorted = n[sort_indices]
    
    for key, pred in predictions.items():
        pred_sorted = pred[sort_indices]
        label = f"{key} Fit (R^2={scores[key]:.4f})"
        plt.plot(n_sorted, pred_sorted, label=label, color=colors[key], linewidth=2)

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
