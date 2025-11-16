# In working_reed_solomon.py, inside GradientCorrector
# NOTE: The return type must change from Tuple[List[int], float, float] to 
# Tuple[List[int], float, float, float].

def _gradient_to_symbols(self, gradient: np.ndarray) -> Tuple[List[int], float, float, float]:
    """Convert gradient to finite field symbols [0, 256] using mean/std scaling."""
    flat = gradient.flatten().astype(np.float32)
    if len(flat) == 0:
        return [], 0.0, 0.0, 0.0

    mean_val = float(np.mean(flat))
    std_val = float(np.std(flat))
    
    # Use 6*std to cover ~99.7% of the data, robust against common outliers.
    scaling_factor = 6.0 
    
    if std_val < 1e-6:
        # Handle near-zero/constant gradients by centering at 128
        symbols = [128] * len(flat)
        return symbols, mean_val, 0.0, 0.0 
    
    scaling_range = scaling_factor * std_val
    
    # Map the gradient to [0, 256] range:
    # scaled = ((flat - mean_val) / scaling_range + 0.5) * 256
    scaled = (flat - mean_val) / scaling_range
    normalized = (scaled + 0.5) * 256
    
    # Round and clip to ensure integers are in [0, 256]
    symbols = [int(np.clip(np.round(x), 0, 256)) for x in normalized]
    
    return symbols, mean_val, std_val, scaling_range
