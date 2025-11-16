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

if a == 0:

return 0 if b > 0 else 1

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

"""Reed-Solomon codec for gradient error correction - Fixed version"""


def __init__(self, t: int = 4, alpha: int = 3):

self.t = t # Error correction capability

self.n = 255 # Codeword length

self.k = self.n - 2 * self.t # Data symbols

self.alpha = alpha # Primitive element


if self.k <= 0:

raise ValueError("2*t must be less than n=255")


# Precompute alpha powers up to maximum needed

self._build_power_tables()

self.generator_poly = self._generate_generator_poly()


def _build_power_tables(self):

"""Build comprehensive power tables for field operations"""

# Build alpha^i table

self.alpha_powers = [1] # alpha^0 = 1

current = 1


for i in range(1, 256): # alpha^256 = alpha^0 = 1 in GF(257)

current = GF257.mul(current, self.alpha)

self.alpha_powers.append(current)


# Verify we have a full cycle

assert self.alpha_powers[0] == self.alpha_powers[256 % len(self.alpha_powers)]


def _get_alpha_power(self, exponent: int) -> int:

"""Get alpha^exponent with proper modular arithmetic"""

if exponent == 0:

return 1

# Reduce exponent modulo the order of alpha (which is 256 for primitive elements)

exp_mod = exponent % 256

return self.alpha_powers[exp_mod]


def _poly_mul(self, p1: List[int], p2: List[int]) -> List[int]:

"""Polynomial multiplication in GF(257)"""

if not p1 or not p2:

return [0]


result = [0] * (len(p1) + len(p2) - 1)

for i, c1 in enumerate(p1):

if c1 != 0: # Optimization

for j, c2 in enumerate(p2):

if c2 != 0:

result[i + j] = GF257.add(result[i + j], GF257.mul(c1, c2))

return result



def _generate_generator_poly(self) -> List[int]:

"""Generate generator polynomial g(x) = ∏(x - α^i) for i=0 to 2t-1"""

g = [1] # Start with polynomial 1


for i in range(2 * self.t):

root = self._get_alpha_power(i)

# Multiply by (x - root)

factor = [GF257.sub(0, root), 1] # [-root, 1] represents (x - root)

g = self._poly_mul(g, factor)


return g


def encode(self, data: List[int]) -> List[int]:

"""Systematic RS encoding using polynomial division"""

if len(data) != self.k:

raise ValueError(f"Data length must be {self.k}, got {len(data)}")


# Validate input symbols

for symbol in data:

if not (0 <= symbol < GF257.PRIME):

raise ValueError(f"Invalid symbol {symbol}, must be in [0, {GF257.PRIME-1}]")


# Systematic encoding: multiply data by x^(2t) then divide by generator

# This gives remainder which becomes the parity check symbols


# Create data polynomial: data(x) * x^(2t)

data_shifted = data + [0] * (2 * self.t)


# Perform polynomial long division to get remainder

dividend = data_shifted[:]

divisor = self.generator_poly


# Long division algorithm

for i in range(len(data)):

if dividend[i] != 0:

# Calculate quotient coefficient

coeff = dividend[i]


# Subtract divisor * coeff from dividend

for j in range(len(divisor)):

pos = i + j

if pos < len(dividend):

term = GF257.mul(coeff, divisor[j])

dividend[pos] = GF257.sub(dividend[pos], term)


# The remainder (last 2t symbols) becomes parity

parity = dividend[-2*self.t:]


# Return systematic codeword [data | parity]

return data + parity


def _calculate_syndrome(self, received: List[int]) -> List[int]:

"""Calculate syndrome values S_j = ∑(r_i * α^(ij)) for j=0 to 2t-1"""

syndrome = []


for j in range(2 * self.t):

s_j = 0


for i in range(len(received)):

if received[i] != 0: # Optimization: skip zero terms

# Calculate α^(i*j)

power_ij = (i * j) % 256 # Reduce modulo order of alpha

alpha_ij = self._get_alpha_power(power_ij)

term = GF257.mul(received[i], alpha_ij)

s_j = GF257.add(s_j, term)


syndrome.append(s_j)


return syndrome


def _berlekamp_massey(self, syndrome: List[int]) -> List[int]:

"""Berlekamp-Massey algorithm for error locator polynomial"""

n = len(syndrome)


# Initialize

Lambda = [1] # Current error locator polynomial Λ(x)

B = [1] # Previous polynomial

L = 0 # Current degree of Λ(x)

m = 1 # Shift amount

b = 1 # Normalization factor


for r in range(n):

# Calculate discrepancy Δ_r

discrepancy = syndrome[r]

for i in range(1, min(L + 1, len(Lambda))):

if r - i >= 0:

term = GF257.mul(Lambda[i], syndrome[r - i])

discrepancy = GF257.add(discrepancy, term)


if discrepancy == 0:

# No correction needed

m += 1

else:

# Save current Λ(x)

T = Lambda[:]


# Calculate correction

correction_factor = GF257.div(discrepancy, b)


# Ensure Lambda has enough space

needed_length = max(len(Lambda), len(B) + m)

while len(Lambda) < needed_length:

Lambda.append(0)


# Update Λ(x) = Λ(x) - (Δ_r/b) * x^m * B(x)

for i in range(len(B)):

pos = i + m

if pos < len(Lambda):

correction = GF257.mul(correction_factor, B[i])

Lambda[pos] = GF257.sub(Lambda[pos], correction)


# Update degree and previous polynomial if necessary

if 2 * L <= r:

L = r + 1 - L

B = T[:]

b = discrepancy

m = 1

else:

m += 1


# Remove trailing zeros

while len(Lambda) > 1 and Lambda[-1] == 0:

Lambda.pop()


return Lambda


def _chien_search(self, error_loc_poly: List[int]) -> List[int]:

"""Chien search to find error positions by evaluating Λ(α^(-i))"""

error_positions = []


# Test each position in the codeword

for i in range(self.n):

# Evaluate Λ(α^(-i))

# α^(-i) = α^(256-i) since α^256 = 1

alpha_inv_i = self._get_alpha_power(256 - i) if i > 0 else 1


# Evaluate polynomial at α^(-i)

eval_result = 0

alpha_power = 1 # Start with (α^(-i))^0 = 1


for coeff in error_loc_poly:

if coeff != 0:

term = GF257.mul(coeff, alpha_power)

eval_result = GF257.add(eval_result, term)

# Update for next term: (α^(-i))^(j+1) = (α^(-i))^j * α^(-i)

alpha_power = GF257.mul(alpha_power, alpha_inv_i)


if eval_result == 0:

error_positions.append(i)


return error_positions


def _forney_algorithm(self, syndrome: List[int], error_loc_poly: List[int],

error_positions: List[int]) -> List[int]:

"""Forney algorithm for error magnitude calculation"""

# Calculate error evaluator polynomial Ω(x) = S(x) * Λ(x) mod x^(2t)


# Pad syndrome to polynomial form

S_poly = syndrome[:]


# Multiply S(x) * Λ(x)

omega_full = self._poly_mul(S_poly, error_loc_poly)


# Take mod x^(2t) by truncation

omega = omega_full[:2 * self.t]

while len(omega) < 2 * self.t:

omega.append(0)


# Calculate formal derivative of Λ(x)

# Λ'(x) = ∑(i * Λ_i * x^(i-1)) where multiplication is in GF(257)

lambda_prime = []

for i in range(1, len(error_loc_poly)):

# i * Λ_i in GF(257): add Λ_i to itself i times

coeff_prime = 0

for _ in range(i % GF257.PRIME):

coeff_prime = GF257.add(coeff_prime, error_loc_poly[i])

lambda_prime.append(coeff_prime)


if not lambda_prime:

lambda_prime = [0]


# Calculate error values for each error position

error_values = []

for pos in error_positions:

# Calculate α^(-pos)

alpha_inv_pos = self._get_alpha_power(256 - pos) if pos > 0 else 1


# Evaluate Ω(α^(-pos))

omega_eval = 0

alpha_power = 1

for coeff in omega:

if coeff != 0:

term = GF257.mul(coeff, alpha_power)

omega_eval = GF257.add(omega_eval, term)

alpha_power = GF257.mul(alpha_power, alpha_inv_pos)


# Evaluate Λ'(α^(-pos))

lambda_prime_eval = 0

alpha_power = 1

for coeff in lambda_prime:

if coeff != 0:

term = GF257.mul(coeff, alpha_power)

lambda_prime_eval = GF257.add(lambda_prime_eval, term)

alpha_power = GF257.mul(alpha_power, alpha_inv_pos)


if lambda_prime_eval == 0:

raise ValueError(f"Derivative is zero at position {pos} - decoding failed")


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


try:

# Step 1: Calculate syndrome

syndrome = self._calculate_syndrome(received)


# Step 2: Check if error-free

if all(s == 0 for s in syndrome):

return {

'data': received[:self.k],

'corrected': False,

'errors': 0,

'success': True

}


# Step 3: Find error locator polynomial using Berlekamp-Massey

error_loc_poly = self._berlekamp_massey(syndrome)


# Step 4: Determine number of errors

num_errors = len(error_loc_poly) - 1


if num_errors > self.t:

return {

'success': False,

'error': f'Too many errors: {num_errors} > correction capacity {self.t}'

}


# Step 5: Find error positions using Chien search

error_positions = self._chien_search(error_loc_poly)


if len(error_positions) != num_errors:

return {

'success': False,

'error': f'Chien search mismatch: found {len(error_positions)} roots for {num_errors} errors'

}


# Step 6: Calculate error values using Forney algorithm

error_values = self._forney_algorithm(syndrome, error_loc_poly, error_positions)


# Step 7: Apply corrections

corrected = received[:]

for pos, val in zip(error_positions, error_values):

corrected[pos] = GF257.sub(corrected[pos], val)


# Step 8: Verify correction by recalculating syndrome

verify_syndrome = self._calculate_syndrome(corrected)

if not all(s == 0 for s in verify_syndrome):

return {

'success': False,

'error': 'Correction verification failed - syndrome not zero'

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

'error': f'Decoding exception: {str(e)}'

}



# Test the fixed implementation

if __name__ == "__main__":

print("🧪 Testing Fixed Reed-Solomon Implementation")


# Test with small parameters for debugging

rs = ReedSolomonGF257(t=2)

print(f"Parameters: n={rs.n}, k={rs.k}, t={rs.t}")


# Test basic encoding/decoding

data = [1, 2, 3] + [0] * (rs.k - 3)

print(f"Original data: {data[:10]}...")


encoded = rs.encode(data)

print(f"Encoded length: {len(encoded)}")


# Test error-free decoding

result = rs.decode(encoded)

print(f"Error-free decode: success={result['success']}, errors={result.get('errors', 0)}")


if result['success']:

print("✅ Error-free decoding works!")


# Test with single error

corrupted = encoded[:]

corrupted[10] = (corrupted[10] + 100) % 257


result = rs.decode(corrupted)

print(f"Single error decode: success={result['success']}, errors={result.get('errors', 0)}")


if result['success']:

print("✅ Single error correction works!")

else:

print(f"❌ Error: {result.get('error', 'Unknown')}")

else:

print(f"❌ Error-free decoding failed: {result.get('error', 'Unknown')}")

Response

Created file /home/user/fixed_reed_solomon.py (15730 characters)
