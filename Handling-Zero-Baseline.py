## 💻 Proposed Code Fix for Line 300

# Original code (causing ZeroDivisionError):
# plateau_reduction = (baseline_stats['plateau_ratio'] - corrected_stats['plateau_ratio']) / baseline_stats['plateau_ratio'] * 100

baseline_ratio = baseline_stats['plateau_ratio']

if baseline_ratio == 0:
    # If the baseline plateau ratio is 0, there's no plateau to reduce.
    # We report 0% reduction, or if the corrected ratio is non-zero (worse), we report 'N/A'.
    if corrected_stats['plateau_ratio'] > 0:
        plateau_reduction = float('inf') # Or handle as a different metric indicating failure
        print("⚠️ Warning: Baseline plateau ratio is zero, but corrected ratio is > 0.")
    else:
        plateau_reduction = 0.0
else:
    # Perform the standard percentage reduction calculation
    plateau_reduction = (baseline_ratio - corrected_stats['plateau_ratio']) / baseline_ratio * 100

print(f"Plateau Reduction: {plateau_reduction:.2f}%")
