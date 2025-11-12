import numpy as np
import pandas as pd

EQUATION_MAP = {
    "O(log n)": "np.log2(n) * 0.002",
    "O(sqrt n)": "np.sqrt(n) * 0.001",
    "O(n)": "n * 0.0005",
    "O(n log n)": "n * np.log2(n) * 0.00005",
    "O(n sqrt n)": "n * np.sqrt(n) * 0.00002",
    "O(n^2)": "(n**2) * 0.00001",
    "O(n^3)": "(n**3) * 0.00000001"
}

def generate_data(complexity, n_samples=100, max_n=1000, noise_factor=0.1):
    """
    Generates a DataFrame with 'InputSize' and 'Time' columns for a given time complexity.
    """
    if complexity not in EQUATION_MAP:
        raise ValueError(f"Unknown complexity: {complexity}")

    n = np.sort(np.random.randint(1, max_n + 1, n_samples))
    expr = EQUATION_MAP[complexity]
    base_time = eval(expr)

    noise = np.random.normal(0, base_time * noise_factor, n_samples)
    time = base_time + np.abs(noise)

    return pd.DataFrame({"InputSize": n, "Time": time})

if __name__ == "__main__":
    """
    Generates and saves CSV datasets for each complexity defined in EQUATION_MAP.
    """
    for key in EQUATION_MAP.keys():
        df = generate_data(key)
        file_name = f"data_on_{key[2:-1].replace(' ', '').replace('^', '').replace('(', '').replace(')', '')}.csv"
        df.to_csv(file_name, index=False)
        print(f"Generated {file_name}")
