import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt

from data_cleaning_improved import *
from data_cleaning import *
from main_algorithm import *
from functions import *
from msm_functions import *
    
def bootstrap_samples_lsh(data, ratio, number_bootstrap, b_r_comb, seed):
    results_lsh = []
    bootstrap, _ = get_bootstrap_samples(data,ratio, number_bootstrap,seed)  

    # Dictionary to accumulate results for each combination of bands and rows
    accumulated_results = {comb: [] for comb in b_r_comb}

    for i, sample in enumerate(bootstrap):
        print(f"Currently on bootstrap {i + 1}")

        true_pairs = get_true_pairs(sample)
        print(f"Number of True Pairs: {len(true_pairs)}")

        binary_matrix = generate_binary_matrix(sample)

        for comb in b_r_comb:

            bands, rows = comb
            print(f"Bands: {bands}, Rows: {rows}")
            
            signature_matrix = generate_signature_matrix(binary_matrix, bands, rows)
            candidate_pairs = generate_candidate_pairs(signature_matrix, bands)
            print(f"True pairs: {len(true_pairs)}")
            print(f"Sample True Pairs: {list(true_pairs)[:5]}")
            print(f"Number of Candidate Pairs: {len(candidate_pairs)}")
            print(f"Sample Candidate Pairs: {list(candidate_pairs)[:5]}")

            # Compute evaluation metrics
            tp = len(candidate_pairs.intersection(true_pairs))  # True positives
            fp = len(candidate_pairs - true_pairs)             # False positives
            fn = len(true_pairs - candidate_pairs)             # False negatives

            pq = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision
            pc = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
            f1_star = (2 * pq * pc) / (pq + pc) if (pq + pc) > 0 else 0
            fraction_comparisons = len(candidate_pairs) / ((len(sample) * (len(sample) - 1)) / 2)

            print(f"TP: {tp}, FP: {fp}, FN: {fn}")
            print(f"PQ: {pq}, PC: {pc}, F1*: {f1_star}")

            t = (1 / bands) ** (1 / rows)

            # Store results
            accumulated_results[comb].append({
                'bands': bands,
                'rows': rows,
                't': t,
                'f1_star': f1_star,
                'pc': pc,
                'pq': pq,
                'fraction_comparisons': fraction_comparisons
            })

    for comb, results in accumulated_results.items():
        print(f"Combination: {comb}, Number of Results: {len(results)}")

    # Calculate averages for each combination of bands and rows
    for comb, results in accumulated_results.items():
        print(f"Combination: {comb}, Results: {results[:2]}")
        avg_f1_star = sum(r['f1_star'] for r in results) / len(results)
        avg_pc = sum(r['pc'] for r in results) / len(results)
        avg_pq = sum(r['pq'] for r in results) / len(results)
        avg_fraction_comparisons = sum(r['fraction_comparisons'] for r in results) / len(results)
        bands, rows = comb

        results_lsh.append({
            'bands': bands,
            'rows': rows,
            't': (1 / bands) ** (1 / rows),
            'f1_star': avg_f1_star,
            'pc': avg_pc,
            'pq': avg_pq,
            'fraction_comparisons': avg_fraction_comparisons
        })

    print("Aggregated Results:")
    for result in results_lsh:
        print(result)

    return results_lsh


def bootstrap_samples_msm(data, ratio, number_bootstrap, b_r_comb, tuning_parameters, alpha, beta, gamma, mu, delta, epsilon_TMWM, seed):
    results_msm = []
    bootstrap, _ = get_bootstrap_samples(data, ratio, number_bootstrap, seed)  # Generate bootstrap samples

    # Dictionary to accumulate results for each combination of bands and rows
    accumulated_results = {comb: [] for comb in b_r_comb}

    for i, sample in enumerate(bootstrap):
        print(f"Currently on bootstrap {i + 1}")

        # Get true pairs for evaluation
        true_pairs = get_true_pairs(sample)
        print(f"Number of True Pairs: {len(true_pairs)}")

        # Generate binary matrix for the sample
        binary_matrix = generate_binary_matrix(sample)

        for comb in b_r_comb:
            bands, rows = comb

            # Create candidate pairs using LSH for the given bands and rows
            signature_matrix = generate_signature_matrix(binary_matrix, bands, rows)
            candidate_pairs = generate_candidate_pairs(signature_matrix, bands)

            print(f"Number of Candidate Pairs for bands {bands} and rows {rows}: {len(candidate_pairs)}")

            # Calculate dissimilarity matrix using MSM
            dissimilarity_matrix = MSM(final_data, candidate_pairs, alpha, beta, gamma, epsilon_TMWM, mu, delta)

            # Tune clusters and evaluate
            result = tuning_cluster(dissimilarity_matrix, true_pairs, candidate_pairs, tuning_parameters)

            # Store results for this bootstrap and combination of bands and rows
            accumulated_results[comb].append({
                'bootstrap': i + 1,
                'bands': bands,
                'rows': rows,
                **result
            })

    # Aggregate results into accumulated_results
    for comb, results in accumulated_results.items():
        avg_f1 = sum(r['F1'] for r in results) / len(results)
        avg_fraction_comparisons = sum(r['fraction_comp'] for r in results) / len(results)
        bands, rows = comb

    # Store aggregated results
        results_msm.append({
            'bands': bands,
            'rows': rows,
            'F1': avg_f1,
            'fraction_comp': avg_fraction_comparisons
        })

    return results_msm