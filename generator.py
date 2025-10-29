import numpy as np
import pandas as pd

def generate_data(complexity, n_samples=100, max_n=1000, noise_factor=0.1):
    """
    Generates a DataFrame with 'InputSize' and 'Time' columns
    for a given time complexity.

    Args:
        complexity (str): 'O(n)', 'O(n^2)', or 'O(n^3)'
        n_samples (int): Number of data points to generate.
        max_n (int): The maximum input size.
        noise_factor (float): How much random noise to add (as a fraction of time).
    """
    print(f"Generating data for {complexity}...")
    
    # Create a non-linear distribution of n values so points aren't just in a straight line
    n = np.sort(np.random.randint(1, max_n + 1, n_samples))
    
    base_time = 0
    
    if complexity == 'O(n)':
        # Linear: time = a * n
        base_time = n * 0.005
    elif complexity == 'O(n^2)':
        # Quadratic: time = a * n^2
        base_time = (n**2) * 0.00001
    elif complexity == 'O(n^3)':
        # Cubic: time = a * n^3
        # We need a very small constant to keep times reasonable
        base_time = (n**3) * 0.00000001
    else:
        raise ValueError("Unknown complexity. Use 'O(n)', 'O(n^2)', or 'O(n^3)'.")

    # Add realistic, proportional noise
    # Noise is scaled by the base_time itself
    noise = np.random.normal(0, base_time * noise_factor, n_samples)
    
    # Ensure time is non-negative
    time = base_time + np.abs(noise)
    
    df = pd.DataFrame({'InputSize': n, 'Time': time})
    return df

if __name__ == "__main__":
    # Generate and save the datasets
    df_n = generate_data('O(n)', max_n=5000)
    df_n.to_csv('data_On.csv', index=False)

    df_n2 = generate_data('O(n^2)', max_n=1000)
    df_n2.to_csv('data_On2.csv', index=False)
    
    df_n3 = generate_data('O(n^3)', max_n=500)
    df_n3.to_csv('data_On3.csv', index=False)

    print("\nGenerated 'data_On.csv', 'data_On2.csv', and 'data_On3.csv'")
    print("You can now run the analysis script on these files, e.g.:")
    print("python analyze_complexity.py data_On2.csv")
