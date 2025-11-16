import numpy as np

from typing import List, Tuple, Dict, Any



class GF257:

"""Finite Field GF(257) operations - mathematically pure"""

PRIME = 257


@staticmethod

def add(a: int, b: int) -> int:

return (a + b) % GF257.PRIME


@staticmethod

def sub(a: int, b: int) -> int:

return (a - b) % GF257.PRIME


@staticmethod

def mul(a: int, b: int) -> int:

return (a * b) % GF257.PRIME


@staticmethod

def power(a: int, b: int) -> int:

"""Fast modular exponentiation"""

result = 1

base = a % GF257.PRIME

while b > 0:

if b & 1:

result = (result * base) % GF257.PRIME

base = (base * base) % GF257.PRIME

b >>= 1

return result



@staticmethod

def inverse(a: int) -> int:

"""Multiplicative inverse using Fermat's Little Theorem"""

if a == 0:

raise ZeroDivisionError("Cannot invert zero in GF(257)")

return GF257.power(a, GF257.PRIME - 2)


@staticmethod

def div(a: int, b: int) -> int:

"""Division a / b = a * b^(-1)"""

return GF257.mul(a, GF257.inverse(b))



class ReedSolomonGF257:

"""Reed-Solomon codec for gradient error correction"""


def __init__(self, t: int = 4, alpha: int = 3):

self.t = t # Error correction capability

self.n = 255 # Codeword length (GF(257) has 256 elements, excluding 0)

self.k = self.n - 2 * self.t # Data symbols

self.alpha = alpha # Primitive element


if self.k <= 0:

raise ValueError("2*t must be less than n=255")


# Precompute powers of alpha

self.alpha_powers = [GF257.power(self.alpha, i) for i in range(self.n)]

self.generator_poly = self._generate_generator_poly()


def _poly_mul(self, p1: List[int], p2: List[int]) -> List[int]:

"""Polynomial multiplication in GF(257)"""

if not p1 or not p2:

return [0]


result = [0] * (len(p1) + len(p2) - 1)

for i, c1 in enumerate(p1):

for j, c2 in enumerate(p2):

result[i + j] = GF257.add(result[i + j], GF257.mul(c1, c2))

return result



def _generate_generator_poly(self) -> List[int]:

"""Generate generator polynomial g(x) = ∏(x - α^i) for i=0 to 2t-1"""

g = [1] # Start with polynomial 1


for i in range(2 * self.t):

root = self.alpha_powers[i]

# Multiply by (x - root)

factor = [GF257.sub(0, root), 1] # [-root, 1]

g = self._poly_mul(g, factor)


return g


def encode(self, data: List[int]) -> List[int]:

"""Systematic RS encoding"""

if len(data) != self.k:

raise ValueError(f"Data length must be {self.k}, got {len(data)}")


# Validate input symbols

for symbol in data:

if not (0 <= symbol < GF257.PRIME):

raise ValueError(f"Invalid symbol {symbol}, must be in [0, {GF257.PRIME-1}]")


# Systematic encoding: [data | parity]

# Calculate remainder of data(x) * x^(2t) divided by g(x)


# Multiply data by x^(2t)

msg_poly = data + [0] * (2 * self.t)


# Polynomial long division

remainder = msg_poly[:]


for i in range(self.k):

if remainder[i] != 0:

# Subtract generator polynomial

for j in range(len(self.generator_poly)):

pos = i + j

if pos < len(remainder):

coeff = GF257.mul(remainder[i], self.generator_poly[j])

remainder[pos] = GF257.sub(remainder[pos], coeff)


# Parity symbols are the last 2t elements of remainder

parity = remainder[-2*self.t:]


return data + parity


def _calculate_syndrome(self, received: List[int]) -> List[int]:

"""Calculate syndrome values S_j = ∑(r_i * α^(ij)) for j=0 to 2t-1"""

syndrome = []


for j in range(2 * self.t):

s_j = 0

alpha_j = self.alpha_powers[j]


for i in range(len(received)):

# received[i] * α^(i*j)

power_ij = (i * j) % (GF257.PRIME - 1) # Exponent modulo order of multiplicative group

alpha_ij = self.alpha_powers[power_ij] if power_ij < len(self.alpha_powers) else GF257.power(self.alpha, power_ij)

term = GF257.mul(received[i], alpha_ij)

s_j = GF257.add(s_j, term)


syndrome.append(s_j)


return syndrome


def _berlekamp_massey(self, syndrome: List[int]) -> List[int]:

"""Berlekamp-Massey algorithm for error locator polynomial"""

n = len(syndrome)

C = [1] # Current error locator polynomial Λ(x)

B = [1] # Previous polynomial

L = 0 # Current degree

m = 1 # Shift

b = 1 # Normalization factor


for r in range(n):

# Calculate discrepancy

discrepancy = syndrome[r]

for i in range(1, min(L + 1, len(C))):

if r - i >= 0:

term = GF257.mul(C[i], syndrome[r - i])

discrepancy = GF257.add(discrepancy, term)


if discrepancy == 0:

m += 1

else:

T = C[:] # Save current C

correction_factor = GF257.div(discrepancy, b)


# Ensure C is long enough

needed_length = max(len(C), len(B) + m)

if len(C) < needed_length:

C.extend([0] * (needed_length - len(C)))


# Update C(x) -= (discrepancy/b) * x^m * B(x)

for i in range(len(B)):

pos = i + m

if pos < len(C):

correction = GF257.mul(correction_factor, B[i])

C[pos] = GF257.sub(C[pos], correction)


# Update L and B if necessary

if 2 * L <= r:

L = r + 1 - L

B = T

b = discrepancy

m = 1

else:

m += 1


# Return polynomial trimmed to actual degree

while len(C) > 1 and C[-1] == 0:

C.pop()


return C


def _chien_search(self, error_loc_poly: List[int]) -> List[int]:

"""Chien search to find error positions"""

error_positions = []


# Check each position in the codeword

for i in range(self.n):

# Evaluate Λ(α^(-i))

alpha_inv_i = GF257.inverse(self.alpha_powers[i % len(self.alpha_powers)])


eval_result = 0

for j, coeff in enumerate(error_loc_poly):

if coeff != 0:

term = GF257.mul(coeff, GF257.power(alpha_inv_i, j))

eval_result = GF257.add(eval_result, term)


if eval_result == 0:

error_positions.append(i)


return error_positions


def _forney_algorithm(self, syndrome: List[int], error_loc_poly: List[int],

error_positions: List[int]) -> List[int]:

"""Forney algorithm for error values"""

# Calculate error evaluator polynomial Ω(x)

# Ω(x) = S(x) * Λ(x) mod x^(2t)


# Extend syndrome to polynomial

S_poly = syndrome + [0] * max(0, len(error_loc_poly) - len(syndrome))


# Multiply S(x) * Λ(x)

omega_full = self._poly_mul(S_poly, error_loc_poly)


# Take mod x^(2t) by truncation

omega = omega_full[:2 * self.t]


# Calculate formal derivative of Λ(x)

lambda_prime = []

for i in range(1, len(error_loc_poly)):

# Formal derivative: coefficient of x^i becomes i * coeff

# In GF(257): i * coeff = coeff + coeff + ... (i times)

coeff_prime = 0

for _ in range(i % GF257.PRIME): # Avoid overflow

coeff_prime = GF257.add(coeff_prime, error_loc_poly[i])

lambda_prime.append(coeff_prime)


if not lambda_prime:

lambda_prime = [0]


# Calculate error values

error_values = []

for pos in error_positions:

alpha_inv_pos = GF257.inverse(self.alpha_powers[pos % len(self.alpha_powers)])


# Evaluate Ω(α^(-pos))

omega_eval = 0

for j, coeff in enumerate(omega):

if coeff != 0:

term = GF257.mul(coeff, GF257.power(alpha_inv_pos, j))

omega_eval = GF257.add(omega_eval, term)


# Evaluate Λ'(α^(-pos))

lambda_prime_eval = 0

for j, coeff in enumerate(lambda_prime):

if coeff != 0:

term = GF257.mul(coeff, GF257.power(alpha_inv_pos, j))

lambda_prime_eval = GF257.add(lambda_prime_eval, term)


if lambda_prime_eval == 0:

raise ValueError(f"Derivative is zero at position {pos}")


# Error value = -Ω(α^(-pos)) / Λ'(α^(-pos))

error_val = GF257.div(GF257.sub(0, omega_eval), lambda_prime_eval)

error_values.append(error_val)


return error_values


def decode(self, received: List[int]) -> Dict[str, Any]:

"""Complete Reed-Solomon decoding with error correction"""

if len(received) != self.n:

return {

'success': False,

'error': f'Received length {len(received)} != codeword length {self.n}'

}


# Calculate syndrome

syndrome = self._calculate_syndrome(received)


# Check if error-free

if all(s == 0 for s in syndrome):

return {

'data': received[:self.k],

'corrected': False,

'errors': 0,

'success': True

}


try:

# Find error locator polynomial

error_loc_poly = self._berlekamp_massey(syndrome)


# Estimate number of errors

num_errors = len(error_loc_poly) - 1


if num_errors > self.t:

return {

'success': False,

'error': f'Too many errors: {num_errors} > {self.t}'

}


# Find error positions

error_positions = self._chien_search(error_loc_poly)


if len(error_positions) != num_errors:

return {

'success': False,

'error': f'Chien search failed: found {len(error_positions)} roots for degree {num_errors} polynomial'

}


# Calculate error values

error_values = self._forney_algorithm(syndrome, error_loc_poly, error_positions)


# Apply corrections

corrected = received[:]

for pos, val in zip(error_positions, error_values):

corrected[pos] = GF257.sub(corrected[pos], val)


# Verify correction

verify_syndrome = self._calculate_syndrome(corrected)

if not all(s == 0 for s in verify_syndrome):

return {

'success': False,

'error': 'Correction verification failed'

}


return {

'data': corrected[:self.k],

'corrected': True,

'errors': num_errors,

'success': True,

'error_positions': error_positions,

'error_values': error_values

}


except Exception as e:

return {

'success': False,

'error': f'Decoding failed: {str(e)}'

}



class GradientCorrector:

"""Apply Reed-Solomon error correction to gradient vectors"""


def __init__(self, rs_codec: ReedSolomonGF257):

self.rs = rs_codec


def correct_gradient(self, gradient: np.ndarray) -> Dict[str, Any]:

"""Apply RS correction to noisy gradient"""

try:

# Convert gradient to GF(257) symbols

gradient_symbols = self._gradient_to_symbols(gradient)


# Pad or truncate to data length

if len(gradient_symbols) > self.rs.k:

gradient_symbols = gradient_symbols[:self.rs.k]

elif len(gradient_symbols) < self.rs.k:

gradient_symbols.extend([0] * (self.rs.k - len(gradient_symbols)))


# Encode with RS

encoded = self.rs.encode(gradient_symbols)


# Simulate noise (for testing)

corrupted = encoded[:]

# Add up to t errors for demonstration

import random

error_count = min(self.rs.t, 2) # Add 2 errors

error_positions = random.sample(range(len(corrupted)), error_count)

for pos in error_positions:

corrupted[pos] = (corrupted[pos] + random.randint(1, 256)) % 257


# Decode and correct

result = self.rs.decode(corrupted)


if result['success']:

corrected_gradient = self._symbols_to_gradient(result['data'], gradient.shape)

return {

'original_gradient': gradient,

'corrected_gradient': corrected_gradient,

'errors_corrected': result['errors'],

'success': True

}

else:

return {

'original_gradient': gradient,

'corrected_gradient': gradient,

'errors_corrected': 0,

'success': False,

'error': result['error']

}


except Exception as e:

return {

'success': False,

'error': f'Gradient correction failed: {str(e)}'

}


def _gradient_to_symbols(self, gradient: np.ndarray) -> List[int]:

"""Convert gradient values to GF(257) symbols"""

# Flatten gradient

flat_grad = gradient.flatten()


# Normalize to [0, 256] range

if len(flat_grad) == 0:

return []


min_val = np.min(flat_grad)

max_val = np.max(flat_grad)


if max_val == min_val:

return [128] * len(flat_grad) # Neutral value


# Scale to [0, 256]

normalized = (flat_grad - min_val) / (max_val - min_val) * 256

symbols = np.clip(normalized, 0, 256).astype(int).tolist()


return symbols


def _symbols_to_gradient(self, symbols: List[int], original_shape: tuple) -> np.ndarray:

"""Convert GF(257) symbols back to gradient shape"""

# For this demo, just return symbols as float32 array

# In practice, you'd need to store normalization parameters

gradient_flat = np.array(symbols, dtype=np.float32)


# Reshape to original shape (pad/truncate if needed)

target_size = np.prod(original_shape)

if len(gradient_flat) > target_size:

gradient_flat = gradient_flat[:target_size]

elif len(gradient_flat) < target_size:

gradient_flat = np.pad(gradient_flat, (0, target_size - len(gradient_flat)))


return gradient_flat.reshape(original_shape)



if __name__ == "__main__":

# Test the Reed-Solomon implementation

print("🧪 Testing Reed-Solomon GF(257) Implementation")


# Create RS codec

rs = ReedSolomonGF257(t=4) # Can correct up to 4 errors

print(f"RS Parameters: n={rs.n}, k={rs.k}, t={rs.t}")


# Test data

test_data = list(range(1, rs.k + 1)) # [1, 2, 3, ..., k]

print(f"Original data: {test_data[:10]}... (length: {len(test_data)})")


# Encode

encoded = rs.encode(test_data)

print(f"Encoded: {encoded[:10]}... (length: {len(encoded)})")


# Test error-free decoding

result = rs.decode(encoded)

print(f"Error-free decode: {result['success']}, errors: {result['errors']}")


# Test with errors

corrupted = encoded[:]

corrupted[5] = (corrupted[5] + 100) % 257 # Add error

corrupted[10] = (corrupted[10] + 200) % 257 # Add another error


result = rs.decode(corrupted)

print(f"With 2 errors - Success: {result['success']}, errors corrected: {result.get('errors', 0)}")


if result['success']:

print(f"Data recovered correctly: {result['data'] == test_data}")


# Test gradient correction

print("\n🧪 Testing Gradient Correction")

corrector = GradientCorrector(rs)


# Create test gradient

gradient = np.random.randn(32, 16) # Typical layer gradient


result = corrector.correct_gradient(gradient)

print(f"Gradient correction - Success: {result['success']}")

print(f"Errors corrected: {result.get('errors_corrected', 0)}")

print(f"Original shape: {gradient.shape}")

print(f"Corrected shape: {result.get('corrected_gradient', np.array([])).shape}")

Response

Created file /home/user/reed_solomon_gf257.py (17701 characters)
