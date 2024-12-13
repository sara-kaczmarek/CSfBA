import pandas as pd
import numpy as np
import re
from strsimpy.qgram import QGram
from textdistance import levenshtein as txtdist

def get_index_by_title(title, df):
    index = df[df['title'] == title].index
    if index.empty:
        raise ValueError(f"Title '{title}' not found in the DataFrame.")
    return index[0]
    
def get_brand_name_by_idx(idx, df):
    feauture_map = get_fmap_by_idx(idx, df)
    brand = get_brand_name_by_fmap(feauture_map)  
    return brand

def get_fmap_by_idx(idx, df):
    features_product = df.iloc[idx]['featuresMap']
    return features_product

def get_brand_name_by_fmap(fmap):
    return fmap.get("brand") or fmap.get("brand name")

def get_shop_by_idx(idx, df):
    if idx is None or idx < 0 or idx >= len(df):
        raise ValueError(f"Invalid index: {idx}. DataFrame has {len(df)} rows.")
    product_shop = df.iloc[idx]["shop"]
    return product_shop

def get_title_by_idx(idx, df):
    p_title = df.iloc[idx]["title"]
    return p_title

def calculate_qgram_similarity_version(string1, string2):
    if string1 is None or string2 is None:
        return 0 
        
    qgram = QGram(3)
    n1 = len((string1))
    n2 = len((string2))
    value_sim = (n1 + n2 - qgram.distance(string1, string2))/ (n1 + n2)
    return value_sim

def split_mw(mw):
    # Use regular expressions to split the model word into numeric and non-numeric parts
    numeric_parts = re.findall(r'\d+\.?\d*', mw)
    non_numeric_parts = re.sub(r'\d+\.?\d*', '', mw)

    # Join numeric parts to form the complete numeric part
    numeric_part = ''.join(numeric_parts)

    return numeric_part, non_numeric_parts

def calculate_cosine_sim(a: str, b: str):
    set_a = set(a.split()) # ---> splits the title model words on the basis of a space, and puts them in a set
    set_b = set(b.split())
    # a and b are sets of elements
    intersection = set_a.intersection(set_b)
    intersection_size = len(intersection)
    return intersection_size / (math.sqrt(len(a)) * math.sqrt(len(b)))

def get_title_mws_by_index(product_index: int, df) -> set: # --> By product index
    return df.iloc[product_index]["title"]

def lv_sim(word_i, word_j) -> float:
    lv_dist = txtdist.levenshtein.normalized_distance(word_i,word_j)
    lv_sim = (1 - lv_dist)
    return lv_sim


def avg_lv_sim(X: set, Y: set) -> float:
    total_sim = 0
    total_length = 0
    for x in X:
        for y in Y:
            total_sim += lv_sim(x, y) * (len(x) + len(y))
            total_length += (len(x) + len(y))
    
    return total_sim / total_length

def calc_sim(string1, string2):
    qgram = QGram(3)
    n1 = len((string1))
    n2 = len((string2))
    value_sim = (n1 + n2 - qgram.distance(string1, string2))/ (n1 + n2)
    return value_sim

def same_shop(idx_i, idx_j, df):
    shop_i = get_shop_by_idx(idx_i, df)
    shop_j = get_shop_by_idx(idx_j, df)
    return shop_i == shop_j

def diff_brand(idx_i, idx_j, df):
    brand_i = get_brand_name_by_idx(idx_i, df)
    brand_j = get_brand_name_by_idx(idx_j, df)
    return brand_i == brand_j

def key(q: tuple) -> str:
    return q[0]

def value(q: tuple) -> str:
    return q[1]

def mw(C: set, D: set) -> float:
    # WIth C & D the set of matching model words
    intersection = C.intersection(D)
    union = C.union(D)
    return len(intersection) / len(union)

def text_distance(non_num_x, non_num_y, threshold: float) -> bool:
    dist = txtdist.levenshtein.normalized_distance(non_num_x, non_num_y)
    sim = 1 - dist
    if sim >= threshold:
            return True
    else:
        return False

def compute_values_sim(pi_idx, pj_idx, df, gamma):
    mk_sim = 0
    total_sim = 0
    matches = 0
    total_weight = 0

    mws_i = df.iloc[pi_idx]["model_words"].split()
    mws_j = df.iloc[pj_idx]["model_words"].split()

    kvp_i = get_fmap_by_idx(pi_idx, df)
    kvp_j = get_fmap_by_idx(pj_idx, df)

    nmk_i = list(kvp_i.keys())
    nmk_j = list(kvp_j.keys())

    for key_i in kvp_i.keys():
        for key_j in kvp_j.keys():  # Corrected typo here from kvp_j.jeys() to kvp_j.keys()
            key_sim = calculate_qgram_similarity_version(key_i, key_j)

            if key_sim > gamma:
                value_sim = calculate_qgram_similarity_version(kvp_i[key_i], kvp_j[key_j])

                total_sim += key_sim * value_sim
                matches += 1
                total_weight += key_sim

                if key_i in nmk_i:
                    nmk_i.remove(key_i)
                if key_j in nmk_j:
                    nmk_j.remove(key_j)

    if total_weight > 0:
        mk_sim = total_sim / total_weight

    nm_mw_i = set()
    for key in nmk_i:
        nm_mw_i.update(ex_mw(kvp_i[key], mws_i))

    nm_mw_j = set()
    for key in nmk_j:
        nm_mw_j.update(ex_mw(kvp_j[key], mws_j))

    if nm_mw_i or nm_mw_j:
        nmk_sim = mw(nm_mw_i, nm_mw_j)
    else:
        nmk_sim = 0

    return mk_sim, nmk_sim, matches

def ex_mw(string, mws):
    extracted_mws = set()
    for mw in mws:
        if mw in string:
            extracted_mws.add(mw)
    return extracted_mws

def TMWA(alpha, beta, delta, eps, pi_idx, pj_idx, df):
    # Step 1: Calculate initial cosine similarity
    a = df.iloc[pi_idx]["title"]
    b = df.iloc[pj_idx]["title"]
    title_cos_sim = calculate_cosine_sim(a, b)
    if title_cos_sim > alpha:
        return 1

    # Step 2: Extract model words
    title_mws_i = get_title_mws_by_index(pi_idx, df).split()
    title_mws_j = get_title_mws_by_index(pj_idx, df).split()

    # Step checking whether they can be identified as different
    for mw_i in title_mws_i:
        for mw_j in title_mws_j:
            num_i, non_num_i = split_mw(mw_i)
            num_j, non_num_j = split_mw(mw_j)

            if text_distance(non_num_i, non_num_j, 1-eps) and num_i != num_j:
                return -1

    # Step 3: Calculate initial similarity based on model words
    final_title_sim = beta * title_cos_sim + (1 - beta) * avg_lv_sim(title_mws_i, title_mws_j)

    # Step 4: Iterate over model words to refine similarity
    found_valid_pair = False  # Track if any valid pair is found
    numerator = 0
    denominator = 0
    for mw_i in title_mws_i:
        for mw_j in title_mws_j:
            num_i, non_num_i = split_mw(mw_i)
            num_j, non_num_j = split_mw(mw_j)
            if text_distance(non_num_i, non_num_j, alpha) and num_i == num_j:
                found_valid_pair = True
                numerator += lv_sim(mw_i, mw_j) * (len(mw_i) + len(mw_j))
                denominator += (len(mw_i) + len(mw_j))

    if found_valid_pair:
        final_title_sim = delta * (numerator / denominator) + (1 - delta) * final_title_sim

    return final_title_sim

    
def get_min_features(idx_i, idx_j, df):
    # Get the feature maps (dicts) for both products
    features_i = get_fmap_by_idx(idx_i, df)
    features_j = get_fmap_by_idx(idx_j, df)
    
    # Calculate the number of features for each
    num_features_i = len(features_i)
    num_features_j = len(features_j)
    
    # Return the minimum
    return min(num_features_i, num_features_j)

def extract_values_to_set(features: dict) -> set:
    value_set = set()
    for value in features.values():
        # Convert value to string if it's not already
        value_str = str(value)
        value_set.add(value_str)
    return value_set  


