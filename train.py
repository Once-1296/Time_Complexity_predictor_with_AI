import pandas as pd
import math
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import json
from utils import helper, DERIVATIVE_ITERATIONS
import pickle

MAP_COMPLEXITY = {}
REVERSE_MAP_COMPLEXITY = {}
COMPLEXITY_ID = 0

def analyse_data(df):
    global DERIVATIVE_ITERATIONS

    # 🔹 Randomly sample up to 100 rows from the dataframe
    df = df.sample(n=min(100, len(df)), random_state=42).reset_index(drop=True)

    lis_t = df['input_time'].tolist()
    lis_n = df['complexity_scale'].tolist()
    derivative_features = []
    for _ in range(DERIVATIVE_ITERATIONS):
        derivatives, lis_t, lis_n = helper(lis_t, lis_n)
        derivative_features.extend(derivatives)  # Flatten for ML
    return derivative_features


def load_data(folder_path):
    global COMPLEXITY_ID, MAP_COMPLEXITY, REVERSE_MAP_COMPLEXITY
    X, y = [], []
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    total_files = len(files)

    for idx, filename in enumerate(files, start=1):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        derivative_features = analyse_data(df)
        complexity = df['expected_tc'].iloc[0]
        if complexity not in MAP_COMPLEXITY:
            MAP_COMPLEXITY[complexity] = COMPLEXITY_ID
            REVERSE_MAP_COMPLEXITY[COMPLEXITY_ID] = complexity
            COMPLEXITY_ID += 1
        X.append(derivative_features)
        y.append(MAP_COMPLEXITY[complexity])

        # Progress display
        print(f"\rProcessing data: {idx / total_files * 100:.1f}%", end="")

    print()  # newline after progress
    # Pad to same length for ML input
    with open("map_data.json", "w") as json_file:
        json.dump(MAP_COMPLEXITY, json_file, indent=4)
    with open("rmap_data.json", "w") as json_file:
        json.dump(REVERSE_MAP_COMPLEXITY, json_file, indent=4)
    max_len = max(len(x) for x in X)
    with open("max_len.txt", "w") as f:
        f.write(str(max_len))
    X_padded = np.array([
    (x * (max_len // len(x) + 1))[:max_len] if len(x) < max_len else x[:max_len]
    for x in X
])
    return X_padded, np.array(y)


def main():
    folder_path = 'train'
    X, y = load_data(folder_path)
    print("Splitting data and training model...")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = DecisionTreeClassifier()

    # Training progress display (simple simulation)
    print("Training progress:")
    for i in range(1, 101, 10):
        print(f"\rTraining: {i}%", end="")
    clf.fit(X_train, y_train)
    print("\rTraining: 100%")

    accuracy = clf.score(X_test, y_test)
    with open('TC_regression.pickle', 'wb') as file:
        pickle.dump(clf, file)
    print(f'Model Accuracy: {accuracy * 100:.2f}%')


if __name__ == '__main__':
    main()
