# ... (Before line 300)
    print(f"With RS - Avg plateau variance: {corrected_stats['avg_plateau_variance']:.2e}")
    
    baseline_ratio = baseline_stats['plateau_ratio']
    
    # FIX: Implement safe division to prevent ZeroDivisionError
    if baseline_ratio > 0:
        plateau_reduction = (baseline_ratio - corrected_stats['plateau_ratio']) / baseline_ratio * 100
    else:
        # If the baseline plateau ratio is 0, we cannot calculate a percentage reduction.
        # We assume 0% reduction if the corrected ratio is also 0 (no change), 
        # or indicate that the baseline was already perfect.
        plateau_reduction = 0.0 
    
    print(f"\nPlateau occurrence reduction: {plateau_reduction:.1f}%")
    # ... (Rest of the file)
