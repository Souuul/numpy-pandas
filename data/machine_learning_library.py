import numpy as np
def numerical_derivative(f, x):
    delta_x = 1e-4
    derivative_x = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'])
    
    while not it.finished:
        idx = it.multi_index
        
        tmp = x[idx]
        
        x[idx]= tmp + delta_x
        fx_plus_delta = f(x)
        
        x[idx]= tmp - delta_x
        fx_minus_delta = f(x)
        
        derivative_x[idx] = (fx_plus_delta - fx_minus_delta) / (2 * delta_x)
        
        x[idx] = tmp
        
        it.iternext()
        
        
    return derivative_x
