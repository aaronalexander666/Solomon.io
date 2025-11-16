# Scale to [0, 256]
        normalized = (flat_grad - min_val) / (max_val - min_val) * 256
        symbols = np.clip(normalized, 0, 256).astype(int).tolist()
