def _calculate_syndrome(self, received: List[int]) -> List[int]:
        """Calculate syndrome vector"""
        syndrome = []
        
        # FIX: Standard RS uses roots alpha^1 to alpha^(2t)
        # Change loop range from j in [0, 2t-1] to j_root in [1, 2t]
        for j_root in range(1, 2 * self.t + 1): 
            s_j = 0
            # Use alpha^(j_root) as the evaluation point
            alpha_j_power = self.gf_pow_fast(self.alpha, j_root) 
            
            for i in range(len(received)):
                if received[i] != 0:
                    # S_j += received[i] * (alpha^j_root)^i
                    power = self.gf_pow_fast(alpha_j_power, i)
                    term = GF257.mul(received[i], power)
                    s_j = GF257.add(s_j, term)
            
            syndrome.append(s_j)
        
        return syndrome
