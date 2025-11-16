# --- GradientCorrector Class in working_reed_solomon.py ---

class GradientCorrector:
    # ... (other methods)
    
    def _tensor_to_symbols(self, tensor: torch.Tensor) -> Tuple[List[int], float, float, float]:
        """Convert PyTorch tensor to finite field symbols [0, 256] using mean/std scaling."""
        flat = tensor.flatten().to(torch.float32)
        if flat.numel() == 0:
            return [], 0.0, 0.0, 0.0

        mean_val = float(flat.mean())
        std_val = float(flat.std())
        
        # Use a scaling factor (e.g., 6*std) to cover ~99.7% of the data if Gaussian.
        # This range [mean - 3*std, mean + 3*std] maps to [0, 256].
        scaling_factor = 6.0 
        
        if std_val < 1e-6:
            # Handle near-zero/constant gradients by centering at 128
            symbols = [128] * flat.numel()
            return symbols, mean_val, 0.0, 0.0 # std_val and scaling_range are 0.0
        
        scaling_range = scaling_factor * std_val
        
        # 1. Scale and center (normalized range [-0.5, 0.5])
        # 2. Shift to [0, 1]: + 0.5
        # 3. Scale to [0, 256]: * 256
        scaled = (flat - mean_val) / scaling_range + 0.5
        normalized = scaled * 256
        
        # Clip and convert to integer symbols (0 to 256)
        symbols = [int(torch.clip(torch.round(x), 0, 256).item()) for x in normalized]
        
        # Return mean, std (for safety), and scaling_range for denormalization
        return symbols, mean_val, std_val, scaling_range

    def _symbols_to_tensor(self, symbols: List[int], shape: torch.Size, dtype: torch.dtype, mean_val: float, std_val: float, scaling_range: float) -> torch.Tensor:
        """Convert symbols back to PyTorch tensor using mean/std scaling"""
        
        arr = torch.tensor(symbols, dtype=torch.float32)

        if std_val < 1e-6:
            # Restore the constant mean value
            denormalized = torch.full(shape, mean_val, dtype=dtype)
        else:
            # Denormalize:
            # 1. Scale to [0, 1]: arr / 256.0
            # 2. Center around 0: (arr / 256.0) - 0.5
            # 3. Restore magnitude: ((arr / 256.0) - 0.5) * scaling_range + mean_val
            
            normalized_float = arr / 256.0
            centered = normalized_float - 0.5
            denormalized = centered * scaling_range + mean_val
            
            # Reshape to original size and restore original dtype
            denormalized = denormalized.reshape(shape).to(dtype)
        
        return denormalized

    # NOTE: You must update the correct_gradient method signature to handle the 
    # four return values from _tensor_to_symbols and pass them to _symbols_to_tensor.
