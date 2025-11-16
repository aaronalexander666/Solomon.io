# Simulate noise (for testing)
            corrupted = encoded[:]
            # Add up to t errors for demonstration
            import random
            error_count = min(self.rs.t, 2)  # Add 2 errors
            error_positions = random.sample(range(len(corrupted)), error_count)
            for pos in error_positions:
                corrupted[pos] = (corrupted[pos] + random.randint(1, 256)) % 257
