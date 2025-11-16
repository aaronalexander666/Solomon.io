def _symbols_to_gradient(self, symbols: List[int], original_shape: tuple) -> np.ndarray:
        """Convert GF(257) symbols back to gradient shape"""
        # For this demo, just return symbols as float32 array
        # In practice, you'd need to store normalization parameters <--- THIS IS THE PROBLEM
        gradient_flat = np.array(symbols, dtype=np.float32)
        # ... rest of the function ...
