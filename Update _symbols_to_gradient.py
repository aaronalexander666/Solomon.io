# In working_reed_solomon.py, inside GradientCorrector
# NOTE: The input signature must change.

def _symbols_to_gradient(self, symbols: List[int], shape: tuple, mean_val: float, std_val: float, scaling_range: float) -> np.ndarray:
    """Convert symbols back to gradient using mean/std scaling"""
    
    arr = np.array(symbols, dtype=np.float32)

    if std_val < 1e-6:
        # Restore the constant mean value
        denormalized = np.full(shape, mean_val, dtype=np.float32).flatten()
    else:
        # Denormalize:
        # 1. Scale to [0, 1]: arr / 256.0
        # 2. Center around 0: (arr / 256.0) - 0.5
        # 3. Restore magnitude: centered * scaling_range + mean_val
        
        normalized_float = arr / 256.0
        centered = normalized_float - 0.5
        denormalized = centered * scaling_range + mean_val
        
    # Reshape and pad/truncate as needed
    target_size = int(np.prod(shape))
    if len(denormalized) > target_size:
        denormalized = denormalized[:target_size]
    elif len(denormalized) < target_size:
        denormalized = np.pad(denormalized, (0, target_size - len(denormalized)))
    
    return denormalized.reshape(shape)
