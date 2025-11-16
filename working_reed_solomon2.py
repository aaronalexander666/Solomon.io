# In working_reed_solomon.py, inside GradientCorrector

    # FIX: Robust Mean/Std Scaling
    def _gradient_to_symbols(self, gradient: np.ndarray) -> Tuple[List[int], float, float, float]:
        """Convert gradient to finite field symbols [0, 256] using mean/std scaling."""
        flat = gradient.flatten().astype(np.float32)
        if len(flat) == 0:
            return [], 0.0, 0.0, 0.0

        mean_val = float(np.mean(flat))
        std_val = float(np.std(flat))
        
        # We use a robust range, e.g., 6*std, to map to [0, 256]
        scaling_factor = 6.0 
        
        if std_val < 1e-6:
            symbols = [128] * len(flat)
            # Must return the calculated mean_val and 0.0 for scaling params
            return symbols, mean_val, 0.0, 0.0 
        
        scaling_range = scaling_factor * std_val
        
        # Map: Gradient -> Centered ([-0.5, 0.5]) -> Scaled ([0, 256])
        scaled = (flat - mean_val) / scaling_range
        normalized = (scaled + 0.5) * 256
        
        symbols = [int(np.clip(np.round(x), 0, 256)) for x in normalized]
        
        # Return the three crucial parameters for denormalization
        return symbols, mean_val, std_val, scaling_range
    
    # FIX: Robust Mean/Std Denormalization
    def _symbols_to_gradient(self, symbols: List[int], shape: tuple, mean_val: float, std_val: float, scaling_range: float) -> np.ndarray:
        """Convert symbols back to gradient using mean/std scaling"""
        
        arr = np.array(symbols, dtype=np.float32)

        if std_val < 1e-6:
            # Restore the constant mean value
            denormalized = np.full(shape, mean_val, dtype=np.float32).flatten()
        else:
            # Denormalize: value = ((arr / 256.0) - 0.5) * scaling_range + mean_val
            normalized_float = arr / 256.0
            centered = normalized_float - 0.5
            denormalized = centered * scaling_range + mean_val
            
        target_size = int(np.prod(shape))
        if len(denormalized) > target_size:
            denormalized = denormalized[:target_size]
        elif len(denormalized) < target_size:
            denormalized = np.pad(denormalized, (0, target_size - len(denormalized)))
        
        return denormalized.reshape(shape)

    # NOTE: The correct_gradient method (not shown) must pass all three scaling parameters: mean_val, std_val, scaling_range
