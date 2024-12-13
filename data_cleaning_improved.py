import re

def drop_url(data):
    
    data = data.drop(['url'], axis=1)
    
    return data

def clean_title(data, patterns:dict):
    
    data['title'] = data['title'].str.lower() # Standardize everything to lower case

    for title_pattern, replacement in patterns.items():
        data['title'] = data['title'].str.replace(title_pattern, replacement, regex=True)
    
    return data

def clean_features(features: dict, patterns: dict) -> dict:
    chars_to_strip = ":/()[]{}"
    cleaned_dict = {}

    for key, value in features.items():
        
        key_str = str(key).lower().strip(chars_to_strip) # Strip and standardize key to lower case
        value_str = str(value).lower() # Standardize value to lower case
        
        # Apply regex patterns to the value
        for pattern, replacement in patterns.items():
            value_str = re.sub(pattern, replacement, value_str)
        
        # Add cleaned key-value pair to the dictionary
        cleaned_dict[key_str] = value_str

    # Clean 'Maximum Resolution' if it exists
    if 'maximum resolution' in cleaned_dict:
        resolution_value = cleaned_dict['maximum resolution']
        # Remove unwanted characters and standardize "x"
        resolution_value = re.sub(r'[^\d\sx]', '', resolution_value)  # Remove non-numeric characters except space and "x"
        resolution_value = re.sub(r'\s*x\s*', 'x', resolution_value)  # Standardize "x" usage
        cleaned_dict['maximum resolution'] = resolution_value

    # Standardize 'Screen Size Class'
    if 'screen size class' not in cleaned_dict or cleaned_dict['screen size class'] == '':
        if 'screen size' in cleaned_dict and cleaned_dict['screen size'] != '':
            cleaned_dict['screen size class'] = cleaned_dict['screen size']
        elif 'screen size (measured diagonally' in cleaned_dict and cleaned_dict['screen size (measured diagonally'] != '':
            cleaned_dict['screen size class'] = cleaned_dict['screen size (measured diagonally']

    if 'screen size class' in cleaned_dict:
        screen_value = cleaned_dict['screen size class']
        screen_value = re.sub(r'(\d+\.?\d*)\s*inch.*', r'\1inch', screen_value)  
        screen_value.strip()
        cleaned_dict['screen size class'] = screen_value

    # Standardize 'Screen Refresh Rate'
    if 'screen refresh rate' not in cleaned_dict or cleaned_dict['screen refresh rate'] == '':
        if 'refresh rate' in cleaned_dict and cleaned_dict['refresh rate'] != '':
            cleaned_dict['screen refresh rate'] = cleaned_dict['refresh rate']

    if 'screen refresh rate' in cleaned_dict:
        refresh_value = cleaned_dict['screen refresh rate']
        refresh_value = re.sub(r'(\d+\.?\d*)\s*hz.*', r'\1hz', refresh_value)  
        refresh_value = re.sub(r'trumotion|clearscn|clear motion rate|', r'', refresh_value)  
        refresh_value.strip()
        cleaned_dict['screen refresh rate'] = refresh_value

    # Standardize 'Recommended Resolution'
    if 'recommended resolution' not in cleaned_dict or cleaned_dict['recommended resolution'] == '':
        if 'vertical resolution' in cleaned_dict and cleaned_dict['vertical resolution'] != '':
            cleaned_dict['recommended resolution'] = cleaned_dict['vertical resolution']

    # Standardize 'Aspect Ratio'
    if 'aspect ratio' in cleaned_dict:
        aspect_value = cleaned_dict['aspect ratio']
        aspect_value = re.sub(r'\b(\d+)\s(\d+)\b', r'\1i\2', aspect_value) # replace special character by i
        aspect_value = re.sub(r'\s.*$', '', aspect_value)
        cleaned_dict['aspect ratio'] = aspect_value

    # Remove keys that are no longer needed
    keys_to_remove = ['screen size', 'screen size (measured diagonally', 'refresh rate','vertical resolution']
    for key in keys_to_remove:
        if key in cleaned_dict:
            del cleaned_dict[key]
    
    return cleaned_dict

def fill_brands(data, brands:list): # Proposed future research: also fill in screen size and refresh rate from the title

    data['brand'] = data['featuresMap'].apply(lambda x: x.get('brand', None))                                                                       
    
    # Function to fill missing brand from the title
    def fill_brand_from_title(row):
        if not row['brand']:  # Check if the brand is missing
            title = row['title']
            for brand in brands:
                if brand in title:
                    row['brand'] = brand
                    break  # Stop once a brand is found
        return row

    # Function to update 'brand' in the 'featuresMap' dictionary
    def update_brand_in_features_map(row):
        if 'featuresMap' in row and isinstance(row['featuresMap'], dict):
            row['featuresMap']['brand'] = row['brand']  # Update brand in the dictionary
        return row

    # Apply the functions
    data = data.apply(fill_brand_from_title, axis=1)  # Fill missing brand from the title
    data = data.apply(update_brand_in_features_map, axis=1)  # Update brand in featuresMap dictionary

    return data

def get_model_words(data, threshold):
    # Determine which features meet the threshold criteria
    feature_counts = data['featuresMap'].apply(lambda x: x.keys() if isinstance(x, dict) else []).explode().value_counts()
    num_rows = len(data)
    selected_features = feature_counts[feature_counts / num_rows >= threshold].index.tolist()

    # Extract only the selected features from the `featuresMap` and remove unselected features
    def filter_features(x):
        if isinstance(x, dict):
            return {k: v for k, v in x.items() if k in selected_features}
        else:
            return {}

    data['featuresMap'] = data['featuresMap'].apply(filter_features)

    # Extract only the selected features from the `featuresMap`
    for feature in selected_features:
        data[feature] = data['featuresMap'].apply(lambda x: x.get(feature, '') if isinstance(x, dict) else '')

    # Extract model words from the title
    data['model_words_title'] = data['title']
    model_word_title_pattern = r'([a-zA-Z0-9]*(([0-9]+[^0-9, ]+)|([^\d, ]+[0-9]+))[a-zA-Z0-9]*)'
    data['model_words_title'] = data['title'].apply(lambda x: ' '.join([i[0] for i in re.findall(model_word_title_pattern, x.lower())]))

    # Combine features into a single column
    data['model_words_features'] = data[selected_features].apply(lambda row: ' '.join(row.astype(str)), axis=1)
    
    # Combine title and feature model words
    data['model_words'] = data['model_words_title'] + ' ' + data['model_words_features']

    # Remove duplicate words
    def remove_duplicates_from_model_words(row):
        if isinstance(row, str):
            words = row.split()
            unique_words = set(words)
            return ' '.join(sorted(unique_words))
        else:
            return ''

    data['model_words'] = data['model_words'].apply(lambda x: remove_duplicates_from_model_words(str(x)))

    return data
    

def clean_data_new(data, patterns:dict, completeness):
    
    # Drop the 'url' column
    data = drop_url(data)
    
    # Clean the titles
    data = clean_title(data,patterns=patterns)
    
    # Clean the featuresMap column using the defined patterns
    data["featuresMap"] = data["featuresMap"].apply(lambda x: clean_features(x, patterns))
    
    # Fill missing brand information and update the featuresMap with it
    data = fill_brands(data,brands=brands)
    
    # Extract model words based on the threshold
    data = get_model_words(data, completeness)
    
    return data

brands = ['lg', 'samsung', 'toshiba', 'sony', 'sunbritetv', 'sharp',
          'vizio', 'sceptre', 'hisense', 'haier', 'supersonic', 'sansui',
          'nec', 'sanyo', 'sigmac', 'panasonic', 'coby', 'philips', 'rca',
          'proscan', 'seiki', 'viewsonic', 'upstar', 'epson', 'affinity',
          'hannspree', 'magnavox', 'compaq', 'westinghouse', 'jvc', 'craig',
          'jvc tv', 'insignia', 'dynex', 'mitsubishi']

patterns = {
    r'newegg\.com|thenerds\.net| - best buy| - thenerds\.net': ' ',  # Remove website names
    r'[\'"”]': 'inch' , r'\s*inches?\b': 'inch',  # Standardize inch
    r'\b(built[\s\-]?in|wi[\s\-]?fi)\b': 'wifi', # Standardize wifi
    r'3 d':'3d', # Standardize 3d
    r'blu ray':'bluray', # Standardize blurey
  #  r'\d:\d': r'\1i\2', # Standardize aspect ratio
    r'[\/\(\)\-\[\]\:\–\—\,\&\+\|]':' ',  # Remove special characters
    r'(?<!\d)\.(?!\d|\w)': ' ', # Remove dot unless it is between 2 numbers
    r'(\d+)(?:\.0)?\s*inch': r'\1inch',  # Standardize screen sizes 
    r'hertz': 'hz', r'\b-\s*hz\b|\s*hz': 'hz',  # Standardize hertz
    r'\bplasma tv\b|\bplasma\b': 'plasma',  # Standardize plasma
    r'ledlcd |led backlit lcd|led backlight lcd': 'led lcd', r'lcdtv':'lcd',  # Standardize display type
    r'hdtv':'hd', # Standardize resolution
    r'\b(tv|class|diagonal|size|measured|screen|series|refurbished|westinghouse|clearscn|with|theater|and|of|four|pairs|glasses|internet||apps|factory re certified)\b': '',  # Remove insignificant words
    r'\b(cinema|theater|slim|glossy|syncmaster|black|grey|gray|gold|sleek|smart|classic|edition|model|year)\b': '', # Remove descriptive words
    r'diag.':' ', # Remove diag.
    r'\b(\w+)\b(?=.*\b\1\b)' : r'', # Remove duplicate alphanumeric strings
    r'\s+': ' ',  # Remove multiple spaces
    r'^\s+|\s+$': ''  # Remove leading and trailing spaces
    }