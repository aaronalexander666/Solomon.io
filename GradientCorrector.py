# In GradientCorrector class:

    def _gradient_to_symbols(self, gradient: np.ndarray) -> Tuple[List[int], float, float]:
        """
        Convert gradient values to GF(257) symbols and return normalization params.
        
        Returns: (symbols_list, min_val, max_val)
        """
        flat_grad = gradient.flatten()
        if len(flat_grad) == 0:
            return [], 0.0, 0.0
        
        min_val = np.min(flat_grad)
        max_val = np.max(flat_grad)
        
        if max_val == min_val:
            # Special case for flat gradient: use neutral value (128)
            symbols = [128] * len(flat_grad)
        else:
            # Scale to [0, 256]
            normalized = (flat_grad - min_val) / (max_val - min_val) * 256
            # Symbols must be in [0, 256]
            symbols = np.clip(normalized, 0, 256).astype(int).tolist()
        
        return symbols, float(min_val), float(max_val)

    def _symbols_to_gradient(self, symbols: List[int], original_shape: tuple, 
                             min_val: float, max_val: float) -> np.ndarray:
        """Convert GF(257) symbols back to gradient shape using normalization params."""
        gradient_flat = np.array(symbols, dtype=np.float64) # Use float64 for precision
        
        # Denormalize from [0, 256] back to original range [min_val, max_val]
        if max_val == min_val:
            # Recreate the flat gradient
            denormalized = np.full_like(gradient_flat, min_val)
        else:
            # Inverse of the normalization: denormalized = (symbols/256) * (max_val - min_val) + min_val
            denormalized = (gradient_flat / 256.0) * (max_val - min_val) + min_val

        # Reshape to original shape
        target_size = np.prod(original_shape)
        if len(denormalized) > target_size:
            denormalized = denormalized[:target_size]
        elif len(denormalized) < target_size:
            denormalized = np.pad(denormalized, (0, target_size - len(denormalized)), constant_values=0)
        
        return denormalized.reshape(original_shape).astype(np.float32)
