import numpy as np
import pandas as pd
import time
ID = 0
from sympy import symbols, sqrt, log, factorial, sympify
#from utils import power_simulation,non_linear_simulation

# Define the variable
x = symbols('x')


# Equation map
MAP_EXTRA_EQUATIONS = {
   "sqrt(n)": "sqrt(x)",
   "n sqrt(n)": "x*sqrt(x)",
   "n sqrt(n) log n": "x*sqrt(x)*log2(x)",
   "log n": "log2(x)",
   "(log n)^2": "(log2(x))**2",
   "n log n": "x*log2(x)",
   "n (log n)^2": "x*((log2(x))**2)",
   "n^2 log n": "log2(x) * (x**2)",
   "2^n": "2**x",
   "3^n": "3**x",
   "n!": "factorial(x)"
}


def evaluate_equation(equation_key: str, x_value: float):
   """
   Evaluates the equation corresponding to 'equation_key' from MAP_EXTRA_EQUATIONS
   for the given x_value. Returns result as float rounded to 3 decimals.
   """
   try:
       # Check if key exists
       if equation_key not in MAP_EXTRA_EQUATIONS:
           raise KeyError(f"Equation '{equation_key}' not found in MAP_EXTRA_EQUATIONS.")
      
       # Get equation string
       equation_str = MAP_EXTRA_EQUATIONS[equation_key]
      
       # Convert 'log2' to Sympy format 'log(x, 2)'
       safe_expr = equation_str.replace("log2(x)", "log(x, 2)")
      
       # Convert string to sympy expression
       expr = sympify(safe_expr, locals={"x": x, "sqrt": sqrt, "log": log, "factorial": factorial})
      
       # Substitute and evaluate
       result = expr.subs(x, x_value).evalf()
      
       # Return as float rounded to 3 decimals
       return round(float(result), 3)
  
   except Exception as e:
       print(f"Error evaluating '{equation_key}' with x={x_value}: {e}")
       return None

def ret_val(degrees,x):
    val = 0
    # start_time = time.time()
    for element in degrees:
       coeff, power = element[0],element[1]
       # print(f"{coeff},{power}")
    #    for _ in range(coeff):
    #        power_simulation(power,x)
       val += coeff*(pow(x,power))
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    return val
def create_polynomial_data(degrees):
   dicto = {
       "input_time":[],
       "complexity_scale":[],
       "expected_tc": ""
   }
   print(degrees)
   for input_x in range(1, 201):
       val = ret_val(degrees,input_x)
       dicto["input_time"].append(input_x)
       dicto["complexity_scale"].append(val)
   power = -1
   for element in degrees:
       coeff,pwr = element[0],element[1]
       # print(f"{coeff},{pwr}")
       if coeff != 0:
           power = pwr
           break
   dicto["expected_tc"] = f"O(n^{power})"
   return dicto if power >= 0 else None
def gen_eqn(curr_deg, lis_so_far):
   global ID
   if curr_deg == -1:
       #print(lis_so_far)
       data = create_polynomial_data(lis_so_far)
       #print(data)
       if data is not None:
           data_to_save = pd.DataFrame(data)
           data_to_save.to_csv(f"train/train_{ID}.csv",index = False)
           print(f"Save to train_{ID}.csv\n")
           ID+=1
       return
   for i in range(0,3):
       lis_nex = lis_so_far
       lis_nex.append([i,curr_deg])
       gen_eqn(curr_deg - 1,lis_nex)
       lis_nex.pop()
def make_non_lin_dat():
   global MAP_EXTRA_EQUATIONS
   dicto = {
       "input_time":[],
       "complexity_scale":[],
       "expected_tc": ""
   }
   id =0
   for key in MAP_EXTRA_EQUATIONS.keys():
       R = 10
       X = 1000
       if key == "2^n" or key == "3^n" or key == "n!":
           X = 10
       for x in range(1,X+1):
           W, B= np.random.randint(1,11),np.random.randint(1,11)
           dicto["input_time"].append(x)
           #Value = non_linear_simulation(W,B,evaluate_equation(key,x))
           Value = W*evaluate_equation(key,x) + B
           dicto["complexity_scale"].append(Value)
           print(f"Progress for {key} : {x}/{X}",end="\r")
       dicto["expected_tc"] = "O("+key+")"
       df = pd.DataFrame(dicto)
       df.to_csv(f"train/np_train_{id}.csv",index=False)
       id+=1
          
def main():
   gen_eqn(3,[])
   make_non_lin_dat()
  
if __name__ == "__main__":
   main()
  

