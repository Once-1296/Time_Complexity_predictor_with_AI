import pickle
import numpy as np
import time
import json
from utils import helper, DERIVATIVE_ITERATIONS
import functions_to_test
TRAINED_MODEL_PATH = 'TC_regression.pickle'
TRAINED_MODEL = None

FUNCTIONS_MAP ={
    "fibonacci_iterative": functions_to_test.fibonacci_iterative,
    "quadratic_function": functions_to_test.quadratic_function,
    "linear_function": functions_to_test.linear_function,
}


def temporary_polynomial(n):
    return 3*n**3 + 2*n**2 + 5*n + 7

def temp_logarithmic(n):
    if n <= 0:
        return 0
    import math
    return math.log(n)
def create_features(function_key):
    global FUNCTIONS_MAP
    lis_t, lis_n =[],[]
    # for i in range(1, 101):
    #     val = temporary_polynomial(i)
    #     lis_n.append(val)
    #     lis_t.append(i)
    for _ in range(1,101):
        start_time = time.time()
        FUNCTIONS_MAP[function_key](_)
        end_time = time.time()
        elapsed_time = end_time - start_time
        lis_n.append(temporary_polynomial(_))
        lis_t.append(_)
    derivative_features = []
    for _ in range(DERIVATIVE_ITERATIONS):
        derivatives, lis_t, lis_n = helper(lis_t, lis_n)
        derivative_features.extend(derivatives)  # Flatten for ML
    X=[derivative_features]
    with open("max_len.txt", "r") as f:
        max_len = int(f.read().strip())
    X_padded = np.array([
    (x * (max_len // len(x) + 1))[:max_len] if len(x) < max_len else x[:max_len]
    for x in X
])
    return X_padded


def check_function_complexity(features):
    global TRAINED_MODEL
    if TRAINED_MODEL is None:
        raise ValueError("Model not loaded. Please run main() to load the trained model.")
    predicted_class = TRAINED_MODEL.predict(features)[0]
    return predicted_class

def main():
    global TRAINED_MODEL,TRAINED_MODEL_PATH,FUNCTIONS_MAP
    with open(TRAINED_MODEL_PATH, 'rb') as file:
        TRAINED_MODEL = pickle.load(file)
    print("Model loaded successfully.")
    with open('rmap_data.json', 'r') as json_file:
        rmap_data = json.load(json_file)
    for key in FUNCTIONS_MAP.keys():
        id = check_function_complexity(create_features(key))
        print(f'Predicted Time Complexity for {key}: {rmap_data[str(id)]}')

if __name__ == "__main__":
    main()