import pandas as pd
import numpy as np
import os
from pathlib import Path

def load_csv_data(csv_path):
    """Load CSV data and extract the relevant metrics."""
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None

def get_benchmark_normalization_ranges(all_data):
    """Calculate normalization ranges for each benchmark based on all observed data."""
    mmlu_deltas = []
    gaia_deltas = []
    bbq_deltas = []
    
    # Collect all delta values across all methods/models/traits
    for data in all_data:
        if data is None:
            continue
        for trait in ['extraversion', 'agreeableness', 'neuroticism', 'openness', 'conscientiousness']:
            mmlu_deltas.append(data['mmlu_deltas'][trait])
            gaia_deltas.append(data['gaia_deltas'][trait])
            bbq_deltas.append(data['bbq_deltas'][trait])
    
    # Calculate ranges for normalization
    mmlu_range = max(mmlu_deltas) - min(mmlu_deltas) if mmlu_deltas else 1.0
    gaia_range = max(gaia_deltas) - min(gaia_deltas) if gaia_deltas else 1.0
    bbq_range = max(bbq_deltas) - min(bbq_deltas) if bbq_deltas else 1.0
    
    # Use standard deviation as alternative normalization
    mmlu_std = np.std(mmlu_deltas) if mmlu_deltas else 1.0
    gaia_std = np.std(gaia_deltas) if gaia_deltas else 1.0
    bbq_std = np.std(bbq_deltas) if bbq_deltas else 1.0
    
    print(f"Normalization Statistics:")
    print(f"MMLU - Range: {mmlu_range:.4f}, Std: {mmlu_std:.4f}")
    print(f"GAIA - Range: {gaia_range:.4f}, Std: {gaia_std:.4f}")
    print(f"BBQ  - Range: {bbq_range:.4f}, Std: {bbq_std:.4f}")
    
    return {
        'mmlu_range': mmlu_range, 'gaia_range': gaia_range, 'bbq_range': bbq_range,
        'mmlu_std': mmlu_std, 'gaia_std': gaia_std, 'bbq_std': bbq_std
    }

def calculate_normalized_stability(delta_values, normalization_factors, method='range_normalized', debug=False, trait_name="", method_name=""):
    """Calculate stability score using proper normalization."""
    if not delta_values or len(delta_values) != 3:
        return 0.0
    
    mmlu_delta, gaia_delta, bbq_delta = delta_values
    
    if debug:
        print(f"\n  DEBUG - {method_name} + {trait_name}:")
        print(f"    Raw deltas: MMLU={mmlu_delta:.6f}, GAIA={gaia_delta:.6f}, BBQ={bbq_delta:.1f}")
        print(f"    Normalization factors: MMLU_range={normalization_factors['mmlu_range']:.4f}, GAIA_range={normalization_factors['gaia_range']:.4f}, BBQ_range={normalization_factors['bbq_range']:.1f}")
    
    if method == 'range_normalized':
        # Normalize by the full range observed in each benchmark
        norm_mmlu = mmlu_delta / normalization_factors['mmlu_range']
        norm_gaia = gaia_delta / normalization_factors['gaia_range'] 
        norm_bbq = bbq_delta / normalization_factors['bbq_range']
    elif method == 'std_normalized':
        # Normalize by standard deviation (z-score style)
        norm_mmlu = mmlu_delta / normalization_factors['mmlu_std']
        norm_gaia = gaia_delta / normalization_factors['gaia_std']
        norm_bbq = bbq_delta / normalization_factors['bbq_std']
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    normalized_deltas = [norm_mmlu, norm_gaia, norm_bbq]
    
    if debug:
        print(f"    Normalized deltas: MMLU={norm_mmlu:.6f}, GAIA={norm_gaia:.6f}, BBQ={norm_bbq:.6f}")
    
    # Calculate consistency (inverse of variance) - how consistent are the deltas
    variance = np.var(normalized_deltas)
    consistency = 1.0 / (1.0 + variance)  # Higher consistency = lower variance
    
    # Calculate disruption (inverse of magnitude) - how much overall disruption
    total_magnitude = np.mean(np.abs(normalized_deltas))
    disruption = 1.0 / (1.0 + total_magnitude)  # Higher disruption = lower magnitude
    
    # Optional composite metric (product of consistency and disruption)
    # Note: This is mathematically questionable as it combines different concepts
    composite_stability = consistency * disruption
    
    if debug:
        print(f"    Variance: {variance:.6f}, Total magnitude: {total_magnitude:.6f}")
        print(f"    Consistency: {consistency:.6f}, Disruption: {disruption:.6f}")
        print(f"    Composite stability: {composite_stability:.6f}")
    
    return {
        'consistency': consistency,
        'disruption': disruption,
        'composite_stability': composite_stability,  # Optional, unverified metric
        'normalized_variance': variance,
        'total_magnitude': total_magnitude,
        'normalized_deltas': normalized_deltas
    }

def extract_benchmark_data(csv_dir, method, model, debug=False):
    """Extract data for a specific method and model across all benchmarks."""
    method_map = {
        'peft': 'PEFT',
        'prompting': 'ICL', 
        'steering': 'Steering'
    }
    
    method_name = method_map.get(method, method)
    
    if debug:
        print(f"\n=== EXTRACTING DATA FOR {method_name.upper()} + {model.upper()} ===")
    
    # Load data for each benchmark
    mmlu_file = f"results_{model}_MMLU_{method}.csv"
    gaia_file = f"results_{model}_GAIA_{method}.csv"
    bbq_file = f"results_{model}_BBQ_{method}.csv"
    
    mmlu_data = load_csv_data(os.path.join(csv_dir, mmlu_file))
    gaia_data = load_csv_data(os.path.join(csv_dir, gaia_file))
    bbq_data = load_csv_data(os.path.join(csv_dir, bbq_file))
    
    if debug:
        print(f"  Loading files: {mmlu_file}, {gaia_file}, {bbq_file}")
        if mmlu_data is not None:
            print(f"  MMLU data shape: {mmlu_data.shape}")
            print(f"  MMLU columns: {list(mmlu_data.columns)}")
        if gaia_data is not None:
            print(f"  GAIA data shape: {gaia_data.shape}")
        if bbq_data is not None:
            print(f"  BBQ data shape: {bbq_data.shape}")
    
    if mmlu_data is None or gaia_data is None or bbq_data is None:
        return None
    
    # Extract baseline and trait values
    traits = ['extraversion', 'agreeableness', 'neuroticism', 'openness', 'conscientiousness']
    
    # MMLU data
    mmlu_rows = mmlu_data[mmlu_data['metric'] == 'Accuracy_Avg']
    if mmlu_rows.empty:
        mmlu_rows = mmlu_data[mmlu_data['metric'] == 'Accuracy']
    if mmlu_rows.empty:
        print(f"Warning: No MMLU accuracy data found in {mmlu_file}")
        return None
    
    mmlu_baseline = mmlu_rows['Baseline'].iloc[0]
    mmlu_deltas = {}
    
    if debug:
        print(f"  MMLU baseline: {mmlu_baseline:.6f}")
        print(f"  MMLU trait values and deltas:")
    
    for trait in traits:
        trait_values = mmlu_rows[trait].iloc[0]
        delta = trait_values - mmlu_baseline
        mmlu_deltas[trait] = delta
        if debug:
            print(f"    {trait}: {trait_values:.6f} - {mmlu_baseline:.6f} = {delta:.6f}")
    
    # GAIA data
    gaia_rows = gaia_data[gaia_data['metric'] == 'Accuracy']
    if gaia_rows.empty:
        print(f"Warning: No GAIA accuracy data found in {gaia_file}")
        return None
        
    gaia_baseline = gaia_rows['Baseline'].iloc[0]
    gaia_deltas = {}
    
    if debug:
        print(f"  GAIA baseline: {gaia_baseline:.6f}")
        print(f"  GAIA trait values and deltas:")
    
    for trait in traits:
        trait_values = gaia_rows[trait].iloc[0]
        delta = trait_values - gaia_baseline
        gaia_deltas[trait] = delta
        if debug:
            print(f"    {trait}: {trait_values:.6f} - {gaia_baseline:.6f} = {delta:.6f}")
    
    # BBQ data
    bbq_rows = bbq_data[bbq_data['metric'] == 'S_AMB']
    if bbq_rows.empty:
        print(f"Warning: No BBQ S_AMB data found in {bbq_file}")
        return None
        
    bbq_baseline = bbq_rows['Baseline'].iloc[0]
    bbq_deltas = {}
    
    if debug:
        print(f"  BBQ baseline: {bbq_baseline:.1f}")
        print(f"  BBQ trait values and deltas:")
    
    for trait in traits:
        trait_values = bbq_rows[trait].iloc[0]
        delta = trait_values - bbq_baseline
        bbq_deltas[trait] = delta
        if debug:
            print(f"    {trait}: {trait_values:.1f} - {bbq_baseline:.1f} = {delta:.1f}")
    
    return {
        'method': method_name,
        'model': model,
        'mmlu_deltas': mmlu_deltas,
        'gaia_deltas': gaia_deltas,
        'bbq_deltas': bbq_deltas
    }

def calculate_all_stability_metrics(debug=False):
    """Calculate stability metrics for all methods, personalities, and combinations."""
    csv_dir = "final_scripts_and_results"
    
    methods = ['peft', 'prompting', 'steering']
    models = ['gemma2', 'llama3']
    
    if debug:
        print("=" * 80)
        print("STABILITY METRICS CALCULATION - DEBUG MODE")
        print("=" * 80)
    
    # First pass: collect all data to determine normalization ranges
    all_data = []
    for method in methods:
        for model in models:
            if method == 'steering' and model == 'llama3':
                continue
            data = extract_benchmark_data(csv_dir, method, model, debug=debug)
            all_data.append(data)
    
    # Calculate normalization factors
    norm_factors = get_benchmark_normalization_ranges(all_data)
    
    # Second pass: calculate stability with proper normalization
    all_results = []
    traits = ['extraversion', 'agreeableness', 'neuroticism', 'openness', 'conscientiousness']
    
    for data in all_data:
        if data is None:
            continue
            
        for trait in traits:
            delta_values = [
                data['mmlu_deltas'][trait],
                data['gaia_deltas'][trait],
                data['bbq_deltas'][trait]
            ]
            
            # Calculate stability using both normalization methods
            range_results = calculate_normalized_stability(delta_values, norm_factors, 'range_normalized', 
                                                         debug=debug, trait_name=trait, method_name=data['method'])
            std_results = calculate_normalized_stability(delta_values, norm_factors, 'std_normalized', 
                                                       debug=debug, trait_name=trait, method_name=data['method'])
            
            all_results.append({
                'method': data['method'],
                'model': data['model'],
                'personality': trait,
                'mmlu_delta': delta_values[0],
                'gaia_delta': delta_values[1],
                'bbq_delta': delta_values[2],
                'norm_mmlu': range_results['normalized_deltas'][0],
                'norm_gaia': range_results['normalized_deltas'][1],
                'norm_bbq': range_results['normalized_deltas'][2],
                'normalized_variance': range_results['normalized_variance'],
                'total_magnitude': range_results['total_magnitude'],
                'consistency': range_results['consistency'],
                'disruption': range_results['disruption'],
                'composite_stability': range_results['composite_stability'],  # Optional metric
                'stability_score_range': range_results['composite_stability'],  # For backward compatibility
                'stability_score_std': std_results['composite_stability'],  # For backward compatibility
                'stability_score': range_results['composite_stability']  # Primary score (composite)
            })
    
    return pd.DataFrame(all_results)

def aggregate_stability_by_level(df):
    """Aggregate stability scores by method, personality, and combination levels."""
    results = []
    
    # Level 1: Method-level stability
    method_stability = df.groupby('method')['stability_score'].mean().sort_values(ascending=False)
    for i, (method, score) in enumerate(method_stability.items(), 1):
        results.append({
            'level': 'Method',
            'category': method,
            'stability_score': score,
            'ranking': i
        })
    
    # Level 2: Personality-level stability  
    personality_stability = df.groupby('personality')['stability_score'].mean().sort_values(ascending=False)
    for i, (personality, score) in enumerate(personality_stability.items(), 1):
        results.append({
            'level': 'Personality',
            'category': personality,
            'stability_score': score,
            'ranking': i
        })
    
    # Level 3: Method+Personality combination stability
    combination_stability = df.groupby(['method', 'personality'])['stability_score'].mean().sort_values(ascending=False)
    for i, ((method, personality), score) in enumerate(combination_stability.items(), 1):
        results.append({
            'level': 'Combination',
            'category': f"{method}+{personality}",
            'stability_score': score,
            'ranking': i
        })
    
    return pd.DataFrame(results)

def print_detailed_analysis(df):
    """Print detailed analysis of the stability scores."""
    print("\nDETAILED NORMALIZED STABILITY ANALYSIS")
    print("=" * 100)
    
    print("\nStability Scores by Method and Trait:")
    print("-" * 80)
    
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        print(f"\n{method}:")
        for _, row in method_data.iterrows():
            print(f"  {row['personality']:>15}: Consistency={row['consistency']:.4f}, "
                  f"Disruption={row['disruption']:.4f}, Composite={row['composite_stability']:.4f}")
            print(f"  {'':>15}  Raw deltas: MMLU={row['mmlu_delta']:>6.3f}, "
                  f"GAIA={row['gaia_delta']:>6.3f}, BBQ={row['bbq_delta']:>6.1f}")
            print(f"  {'':>15}  Norm deltas: MMLU={row['norm_mmlu']:>6.3f}, "
                  f"GAIA={row['norm_gaia']:>6.3f}, BBQ={row['norm_bbq']:>6.3f}")

def main(debug=False):
    """Main execution function."""
    print("Calculating normalized stability metrics for personality manipulation methods...")
    
    if debug:
        print("DEBUG MODE ENABLED - Detailed output will be shown")
    
    # Calculate stability for all combinations
    df = calculate_all_stability_metrics(debug=debug)
    
    if df.empty:
        print("No data found. Check CSV file paths.")
        return
    
    # Print detailed analysis
    print_detailed_analysis(df)
    
    # Aggregate by levels
    results_df = aggregate_stability_by_level(df)
    
    # Save results
    output_file = "normalized_stability_results.csv"
    results_df.to_csv(output_file, index=False)
    
    detailed_output = "detailed_normalized_stability.csv"
    df.to_csv(detailed_output, index=False)
    
    print(f"\nAnalysis complete! Results saved to {output_file}")
    print(f"Detailed results saved to {detailed_output}")
    
    print("\nNOTE: Composite stability scores combine consistency and disruption metrics.")
    print("Consistency measures how consistent deltas are across benchmarks (higher = more consistent).")
    print("Disruption measures overall performance impact (higher = less disruptive).")
    print("Composite = Consistency × Disruption (mathematically questionable but provided for comparison).")
    
    print("\nTop 3 Most Stable Methods (by Composite Score):")
    method_results = results_df[results_df['level'] == 'Method'].head(3)
    for _, row in method_results.iterrows():
        print(f"  {row['ranking']}. {row['category']}: {row['stability_score']:.4f}")
    
    print("\nTop 3 Most Stable Personalities:")
    personality_results = results_df[results_df['level'] == 'Personality'].head(3)
    for _, row in personality_results.iterrows():
        print(f"  {row['ranking']}. {row['category']}: {row['stability_score']:.4f}")
    
    print("\nTop 3 Most Stable Combinations:")
    combination_results = results_df[results_df['level'] == 'Combination'].head(3)
    for _, row in combination_results.iterrows():
        print(f"  {row['ranking']}. {row['category']}: {row['stability_score']:.4f}")
    
    print("\nBottom 3 Least Stable Combinations:")
    least_stable = results_df[results_df['level'] == 'Combination'].tail(3)
    for _, row in least_stable.iterrows():
        print(f"  {row['ranking']}. {row['category']}: {row['stability_score']:.4f}")

if __name__ == "__main__":
    import sys
    debug_mode = '--debug' in sys.argv
    main(debug=debug_mode)