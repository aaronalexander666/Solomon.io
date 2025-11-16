# --- Change 1: Update the return signature of _tensor_to_symbols ---
def _tensor_to_symbols(self, tensor: torch.Tensor) -> Tuple[List[int], float, float, float]:
    """Convert PyTorch tensor to finite field symbols [0, 256] using mean/std scaling."""
    flat = tensor.flatten().to(torch.float32)
    if flat.numel() == 0: return [], 0.0, 0.0, 0.0

    mean_val = float(flat.mean())
    std_val = float(flat.std())
    
    # Use robust scaling: scale by std and shift by mean
    # We choose a scaling factor (e.g., 6*std) to cover 99.7% of the data if Gaussian
    scaling_factor = 6.0 
    
    if std_val < 1e-6:
        # Handle near-zero gradients (already near the optimum/barren plateau)
        symbols = [128] * flat.numel()
        return symbols, mean_val, 0.0, 0.0 # Return 0.0 for std_val and scaling_range
    
    # Range used for quantization: [mean - 3*std, mean + 3*std]
    # This range maps to [0, 256]
    scaling_range = scaling_factor * std_val
    
    # Map the gradient to [0, 256] range:
    # 1. Gradient -> Normalized (centered around 0): (flat - mean_val) / scaling_range
    # 2. Normalized -> Scaled to [0, 1]: (flat - mean_val) / scaling_range + 0.5
    # 3. Scaled to [0, 256]: ((flat - mean_val) / scaling_range + 0.5) * 256
    
    scaled = (flat - mean_val) / scaling_range + 0.5
    normalized = scaled * 256
    
    # Clip and convert to integer symbols
    symbols = [int(torch.clip(torch.round(x), 0, 256).item()) for x in normalized]
    
    # Return mean, std, and scaling_range for denormalization
    return symbols, mean_val, std_val, scaling_range


# --- Change 2: Update _symbols_to_tensor to use mean/std ---
def _symbols_to_tensor(self, symbols: List[int], shape: torch.Size, dtype: torch.dtype, mean_val: float, std_val: float, scaling_range: float) -> torch.Tensor:
    """Convert symbols back to PyTorch tensor using mean/std scaling"""
    arr = torch.tensor(symbols, dtype=torch.float32)

    if std_val < 1e-6:
        # Restore the constant mean value
        denormalized = torch.full_like(arr, mean_val)
    else:
        # Denormalize:
        # 1. Scale to [0, 1]: arr / 256.0
        # 2. Center around 0: (arr / 256.0) - 0.5
        # 3. Restore magnitude: ((arr / 256.0) - 0.5) * scaling_range + mean_val
        
        normalized_float = arr / 256.0
        centered = normalized_float - 0.5
        denormalized = centered * scaling_range + mean_val
    
    result_tensor = denormalized.reshape(shape).to(dtype)
    return result_tensor


# --- Change 3: Update correct_gradient call site ---
def correct_gradient(self, gradient: torch.Tensor) -> Dict[str, Any]:
    # ... inside try block
    # 1. Convert PyTorch Tensor to RS Symbols
    # UPDATE: Capture new return variables
    symbols, mean_val, std_val, scaling_range = self._tensor_to_symbols(gradient)
    
    # ... (rest of encoding/decoding logic remains the same)

    if result['success']:
        # 5. Convert corrected symbols back to PyTorch Tensor
        # UPDATE: Pass new scaling variables
        corrected_gradient = self._symbols_to_tensor(
            corrected_gradient_symbols, 
            original_shape, 
            original_dtype, 
            mean_val, 
            std_val,
            scaling_range # NEW ARGUMENT
        )
    # ... (rest of the method)
