def encode(self, data: List[int]) -> List[int]:
        """Encode data using Reed-Solomon code"""
        # ... (error checks)
        
        # Calculate parity symbols
        parity = [0] * (2 * self.t)
        
        # FIX: The evaluation points for parity generation must match the syndrome roots (alpha^1 to alpha^2t)
        for j_root in range(1, 2 * self.t + 1):
            
            # This is parity index relative to the parity array (0 to 2t-1)
            parity_index = j_root - 1 
            
            # For each data symbol, update parity
            for i in range(self.k):
                if data[i] != 0:
                    # Parity[j-1] += data[i] * (alpha^j_root)^i
                    alpha_power = self.gf_pow_fast(self.alpha, (i * j_root) % 256)
                    contribution = GF257.mul(data[i], alpha_power)
                    parity[parity_index] = GF257.add(parity[parity_index], contribution)
        
        return data + parity
