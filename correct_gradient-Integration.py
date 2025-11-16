# In working_reed_solomon.py, inside GradientCorrector

def correct_gradient(self, gradient: np.ndarray) -> Dict[str, Any]:
    # ...
    try:
        # Capture the new scaling parameters
        symbols, mean_val, std_val, scaling_range = self._gradient_to_symbols(gradient)
        
        # ... (rest of encoding/decoding remains the same)
        
        if result['success']:
            # Convert back to gradient using the new parameters
            corrected_gradient = self._symbols_to_gradient(
                result['data'], 
                gradient.shape,
                mean_val,       # NEW ARG
                std_val,        # NEW ARG
                scaling_range   # NEW ARG
            )
            # ... (rest of success block)
