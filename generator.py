import numpy as np
import pandas as pd

EQUATION_MAP = {
    "O(log n)": "np.log2(n)",
    "O(sqrt n)": "np.sqrt(n)",
    "O(n)": "n",
    "O(n log n)": "n * np.log2(n)",
    "O(n sqrt n)": "n * np.sqrt(n)",
    "O(n^2)": "n**2",
    "O(n^3)": "n**3"
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
    base_time = base_time * (0.001 / np.max(base_time))

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
