import numpy as np  
import pandas as pd  
import hashlib
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
from itertools import combinations
from msm_functions import *

def generate_binary_matrix(data):
    # Extract unique words from model words
    unique_words = list(set(word for row in data['model_words'] for word in row.split()))
    binary_matrix = pd.DataFrame(0, columns=unique_words, index=data.index)
    
    # Fill the binary matrix
    for unique_word in unique_words:
        binary_matrix[unique_word] = data['model_words'].apply(lambda x: 1 if unique_word in x else 0)
    
    binary_matrix = binary_matrix.transpose()
    binary_matrix = binary_matrix.to_numpy()
    
    return binary_matrix

def generate_signature_matrix(binary_data, bands, rows, seed = 42):
    permutations = bands * rows  # Number of permutations
    words, signatures = binary_data.shape
    
    sig_matrix = np.full((permutations, signatures), np.iinfo(np.int32).max, dtype=np.int32) # Initialize signature matrix 

    # Set the random seed
    np.random.seed(seed)
    
    # Generate random permutations for MinHashing
    def generate_permutation_indices(words):
        sequence = np.arange(1, words + 1)
        np.random.shuffle(sequence)
        return sequence
    
    for i in range(permutations):  # For each permutation
        signature = np.full((1, signatures), np.inf)
        permutation = generate_permutation_indices(words)
        
        for row in range(words):  # For each row in binary data
            nonzero_indices = np.where(binary_data[row, :] == 1)[0]
            if len(nonzero_indices) == 0:
                continue
            for col in nonzero_indices:  # Update the signature if the permuted value is smaller
                if signature[0, col] > permutation[row]:
                    signature[0, col] = permutation[row]
                    
        sig_matrix[i, :] = signature.astype(int)

    return sig_matrix

def generate_candidate_pairs(signature_matrix, bands, seed=42):

    signature_matrix = pd.DataFrame(signature_matrix)
    candidates = []

    # Set the random seed for reproducibility
    np.random.seed(seed)

    # Split the signature matrix into bands
    rows_per_band = len(signature_matrix) // bands
    for band_index in range(bands):
        start_row = band_index * rows_per_band
        end_row = start_row + rows_per_band if band_index < bands - 1 else len(signature_matrix)
        band = signature_matrix.iloc[start_row:end_row]

        hash_buckets = {}

        for column in band.columns:
            # Create a hashable string representation of the column
            hashed_values = ''.join(str(int(x)) for x in band[column])

            # Use hashlib to generate a deterministic hash
            bucket_hash = int(hashlib.sha256(hashed_values.encode('utf-8')).hexdigest(), 16)

            # Add columns with the same hash to the same bucket
            if bucket_hash not in hash_buckets:
                hash_buckets[bucket_hash] = []
            hash_buckets[bucket_hash].append(column)

        # Generate candidate pairs from the buckets
        for bucket_columns in hash_buckets.values():
            for i in range(len(bucket_columns) - 1):
                for j in range(i + 1, len(bucket_columns)):
                    candidates.append((bucket_columns[i], bucket_columns[j]))

    # Remove duplicate candidate pairs
    candidate_pairs = set(candidates)
    
    return candidate_pairs

def get_true_pairs(data):
    true_duplicates = defaultdict(list)
    
    for idx, product in data.iterrows():
        model_id = product['modelID']
        true_duplicates[model_id].append(idx)
    
    true_pairs = set()
    for indices in true_duplicates.values():
        if len(indices) > 1:
            for pair in combinations(indices, 2):
                true_pairs.add(tuple(sorted(pair)))
    
    return true_pairs

def MSM(df, cand_pairs, alpha, beta, gamma, eps_tmwa, mu, delta):
    n = df.shape[0]
    dist_matrix = np.full((n, n), 1)

    for pair in cand_pairs:
        pi_idx, pj_idx = pair 

        shop_name_i = df.iloc[pi_idx]['shop']
        shop_name_j = df.iloc[pj_idx]['shop']
        brand_name_i = df.iloc[pi_idx]['brand']
        brand_name_j = df.iloc[pj_idx]['brand']

        if shop_name_i != shop_name_j and brand_name_i == brand_name_j:

            mk_sim, nmk_sim, num_matches = compute_values_sim(pi_idx, pj_idx, df, gamma)

            tmwm_sim = TMWA(alpha, beta, delta, eps_tmwa, pi_idx, pj_idx, df)

            min_feat = get_min_features(pi_idx, pj_idx, df)

            if min_feat != 0 and tmwm_sim != 1:
                theta_1 = (1 - mu) * (num_matches / min_feat)
                theta_2 = 1 - mu - theta_1
                final_sim = theta_1 * mk_sim + theta_2 * nmk_sim + mu * tmwm_sim
            elif min_feat != 0 and tmwm_sim == 1:
                theta_1 = num_matches / min_feat
                theta_2 = 1 - theta_1
                final_sim = theta_1 * mk_sim + theta_2 * nmk_sim
            else:
                final_sim = mk_sim

            dist_matrix[pi_idx][pj_idx] = 1 - final_sim
            dist_matrix[pj_idx][pi_idx] = 1 - final_sim

    return dist_matrix

def F1(precision, recall):
    if precision + recall == 0:
        print(f"recall is {recall} and precision = {precision}")
        return 0
    else:
        f1 = (2 * precision * recall) / (precision + recall)
        return f1


def h_clustering(dissimilarity_matrix, eps, candidate_pairs_set):
    # Apply hierarchical clustering
    model = AgglomerativeClustering(n_clusters=None, linkage='complete', distance_threshold=eps, metric = 'precomputed')
    cluster_labels = model.fit_predict(dissimilarity_matrix)

    # Group indices by cluster labels
    cluster_dict = defaultdict(list)
    for idx, label in enumerate(cluster_labels):
        cluster_dict[label].append(idx)

    # Generate valid pairs within candidate pairs only
    pairs = []
    for indices in cluster_dict.values():
        if len(indices) > 1:  # Only process clusters with more than one element
            cluster_pairs = combinations(indices, 2)
            pairs.extend(p for p in cluster_pairs if tuple(sorted(p)) in candidate_pairs_set)
            # Debug: Check filtering
      #      print(f"Cluster indices: {indices}")

    return pairs


def tuning_cluster(dissimilarity_matrix, true_pairs, candidate_pairs, tuning_parameters):
    
    # Precompute all candidate pairs as a set for quick lookups
    candidate_pairs_set = set(map(tuple, candidate_pairs))

    savelist = {}
    f1_scores = []
    for para in tuning_parameters:
        pairs_msm = h_clustering(dissimilarity_matrix, para, candidate_pairs_set)
        
        # Calculate TP, TN, FP, FN
        set_candidate_lsh = set(map(tuple, candidate_pairs))
        set_true_duplicates = set(map(tuple, true_pairs))
        possible_true_pairs = set_candidate_lsh.intersection(set_true_duplicates)

        candidate_pairs_msm = set(map(tuple, pairs_msm))

        
        npair = len(candidate_pairs_msm)
        all_pairs = set(combinations(range(dissimilarity_matrix.shape[0]), 2))
        non_duplicate_pairs = all_pairs - set_true_duplicates
        
        TP = len(candidate_pairs_msm.intersection(possible_true_pairs))
        FP = len(candidate_pairs_msm) - TP
        FN = len(possible_true_pairs) - TP

        precision_PQ = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall_PC = TP / (TP + FN) if (TP + FN) > 0 else 0
        F1_value = F1(precision_PQ, recall_PC)

        fraction_comp = len(candidate_pairs) / len(all_pairs)
        savelist[F1_value] = [precision_PQ, recall_PC, fraction_comp, para, npair]
        f1_scores.append(F1_value)

    f1_score = max(f1_scores)
    rest = savelist[f1_score]

    return {
        'precision': rest[0],
        'recall': rest[1],
        'F1': f1_score,
        'fraction_comp': rest[2],
        'threshold': rest[3],
        'number of pairs': rest[4],
    }
