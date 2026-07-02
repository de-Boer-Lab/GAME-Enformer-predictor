# enformer_utils.py
import math
import requests
# Function to handle Evaluator request
# Fed into the model by Predictor

MATCHER_NULL_RESPONSE = "NULL"

def slice_prediction_tracks(full_track, original_seq_len, model_input_len, bin_size, buffer_bp=320*2*128):
    
    """
    Sliced a full prediction track to keep only bins corresponding to with just
    N-padding predictions removed.
    This function is for predictions that are shorter than the prediction window and are centered in the receptive field.
    Args:
        full_track (np.array): The 3D full-length prediction array from the model (1, 896, num_tracks)
        original_seq_len (int): The length of original, unpadded sequence.
        model_input_len (int): The length of the sequence required by the model
                               after padding/ trimming.
        bin_size (int): The size of each prediction bin in base pairs.
        buffer_bp (int): The total number of base pairs cropped by the model. 
        
    Returns:
        sliced_track (np.array): The sliced 3D prediction track with bins with just
                                 N-padding removed.
        N_in_left_bin (int): Number of N-padding in the leftmost bin with sequence.
        
    """

    # Calculate the total padding
    total_padding = model_input_len - original_seq_len

    left_buffer = buffer_bp//2

    # Sequence is centred but padding is right-biased (Extra N on the right, if total padding is odd)
    left_padding = total_padding // 2
    left_padding_after_buffer = max(0, left_padding - left_buffer)

    # Calculate the indices of bins containing the sequence
    start_bin_index = left_padding_after_buffer // bin_size
    end_bin_index = math.ceil((left_padding_after_buffer + original_seq_len)/bin_size)

    N_in_left_bin = left_padding_after_buffer - (start_bin_index * bin_size) # This calculates the actual number of N bases in the leftmost bin by determining how many bases of padding are left after accounting for the full bins that come before it. This is more accurate than the original logic, especially when the padding does not perfectly align with the bin boundaries.
    

    sliced_track = full_track[:, start_bin_index:end_bin_index, :]

    return sliced_track, N_in_left_bin


def slice_prediction_tracks_for_range(full_track, original_seq_len, range_start, range_end, model_input_len, bin_size, buffer_bp=320*2*128):
    """
    Slice prediction track to keep only bins corresponding to the prediction range.
    
    Args:
        full_track (np.array): The 3D full-length prediction array (1, 896, num_tracks)
        original_seq_len (int): The length of original, unpadded sequence.
        range_start (int): Start position of prediction range in the subsetted sequence
        range_end (int): End position of prediction range in the subsetted sequence
        model_input_len (int): The length of the sequence required by the model
                        after padding/ trimming.
        bin_size (int): The size of each prediction bin in base pairs (128)
        buffer_bp (int): The total number of base pairs cropped by the model
    
    Returns:
        sliced_track (np.array): Prediction track for only the requested range
        extraBases_in_left_bin (int): Number of bases in the leftmost bin that are outside the requested range (i.e. bases that are in the bin but not in the range)
    """
    # Calculate the total padding
    total_padding = model_input_len - original_seq_len # 196608 - 257 = 196351

    left_buffer = buffer_bp//2 # 40960

    # Sequence is centred but padding is right-biased (Extra N on the right, if total padding is odd)
    left_padding = total_padding // 2 # 196351 // 2 = 98175
    # left_padding_after_buffer = max(0, left_padding - left_buffer) # 98175 - 40960 = 57215

    #Adjust range coordinates with the padding
    range_start_padded = range_start + left_padding # NOTE: shouldn't this be left_padding_after_buffer? # 
    range_end_padded = range_end + left_padding # NOTE: shouldn't this be left_padding_after_buffer? It doesn't matter since is is only called for certain situations but will use that in Borzoi

    #Account for model's left buffer which isn't included in the range calculation
    range_start_after_buffer = range_start_padded - left_buffer # NOTE: 
    range_end_after_buffer = range_end_padded - left_buffer
    # range_start_after_buffer = range_start + left_padding - left_buffer
    # range_end_after_buffer = range_end + left_padding - left_buffer

    # Calculate the indices of bins corresponding to the ranges
    start_bin_index = range_start_after_buffer // bin_size
    print("TEST ranges")
    print(start_bin_index)
    
    end_bin_index = math.ceil(range_end_after_buffer/bin_size)
    print(end_bin_index)
    sliced_track = full_track[:, start_bin_index:end_bin_index, :]
    
    # Calculate how much of the range is NOT covered by bins
    first_bin_start = start_bin_index * bin_size - (range_start_padded - left_buffer)
    print(f"First bin start: {first_bin_start}")
    last_bin_end = end_bin_index * bin_size - (range_start_padded - left_buffer)
    print(f"Last bin end: {last_bin_end}")
    
    # Bases outside your range in edge bins
    
    extraBases_in_left_bin = max(0, -first_bin_start)  # Negative means bin starts before range
    extraBases_in_right_bin = max(0, last_bin_end - (range_end - range_start))  # Bin extends past range
    # extraBases_in_left_bin_myway = range_start_after_buffer - (start_bin_index * bin_size) # This calculates the actual number of bases in the leftmost bin that are outside the range by determining how many bases of the range are left after accounting for the full bins that come before it. 
    print("extra bases in left bin")
    print(extraBases_in_left_bin)
    return sliced_track, extraBases_in_left_bin #, extraBases_in_left_bin_myway


def filter_evaluator_request(simplified_targets_df, request_type, cell_type, matcher_ip, matcher_port, molecule=None):
    
    """
    Filters evaluator request based on assay type, cell type, and molecule and calls Matcher if no exact matches are found
    
    Args:
        simplified_targets_df (pd.DataFrame): DataFrame containing simplified target data.
        request_type (str): Requested type of prediction:
            - "accessibility": Uses ATAC and DNASE (concatenated)
            - "expression" Uses CAGE
            - "expression_pol2": Uses CAGE
            - "binding_{molecule}": Uses CHIP assay with specified molecule.
        cell_type (str): Requested cell type for prediction.
        matcher_ip (str): The IP address of the Matcher server.
        matcher_port (int): The port number of the Matcher server.
        molecule (str, optional): TF binding/ histone modification molecule for ChIP-Seq requests.
        
    Returns:
        tuple: A tuple containing four elements:
            - pd.DataFrame or str: A DataFrame of filtered tracks or an error string.
            - str or None: The actual cell type used (requested or matched).
            - list or None: The actual assay type used (requested or matched; e.g. ["CAGE"] or ["ATAC", "DNASE"]).
            - str: The version of the matcher service used, or "N/A".
    """
    request_error_msg = f"Request Error: No requested tracks in the requested type: {request_type} and cell type: {cell_type} found."
    
    # Only connect to the Matcher if needed by this multi-task model
    # NOTE: for the heatmap, for now, Matcher address is REQUIRED. Later, it will be optional
    try:
        matcher_url = f"http://{matcher_ip}:{matcher_port}"

        print(f"Received evaluator request from Predictor to filter and map desired tracks\
            \n Type Requested: {request_type},\
            \n Cell Type: {cell_type}")
        # Normalize inputs to lowercase for case-insensitive handling
        request_type = request_type.lower() if request_type else None
        cell_type = cell_type.lower() if cell_type else None

        # Define TF binding/ histone modification molecule for ChIP-Seq
        molecule = request_type.split("_")[1] if request_type.startswith("binding_") else None
        molecule = molecule.lower() if molecule else None
        print(f"TF Binding/ Histone Modification (if any, else None): {molecule}")

        # 1. Accessibility (Parse both, ATAC and DNASE, tracks and concatenate)
        #check for exact cell_type match in both ATAC-seq or DNase
        if request_type == "accessibility":
            print(f"Parsing both ATAC and DNASE tracks for cell type provided: {cell_type}")
            accessibility_tracks = simplified_targets_df[
                (simplified_targets_df['Assay'].isin(['ATAC', 'DNASE'])) &
                (simplified_targets_df['Cell Type'].str.lower() == cell_type)
            ]
            # if an exact match was found that the cell_type_actual = cell_type_requested 
            if not accessibility_tracks.empty:
                request_actual = accessibility_tracks['Assay'].unique().tolist()
                return accessibility_tracks, cell_type, request_actual, "N/A"

            # if no exact match was found use the Matcher module to find a closely related cell type
            if accessibility_tracks.empty:
                print(f"No exact matching cell types in ATAC-seq/DNAse assay for cell type: {cell_type}. Querying Matcher for similar cell types in ATAC and DNASE tracks.")
                
                # NOTE: Send request to Matcher here -- We only care about cell-type matching at the moment for the heatmap
                # Binding_{molecule} (ChIP-seq) will be prioritized after first heatmap is generated.
                # Filter out all accessibility tracks -- ATAC and DNAse
                accessibility_tracks_all = simplified_targets_df[(simplified_targets_df['Assay'].isin(['ATAC', 'DNASE']))]
                # set up any dictionary to send to matcher
                message_for_Matcher = {
                    'cell_type_requested': cell_type,
                    'cell_type_list': accessibility_tracks_all['Cell Type'].unique().tolist()
                    }
                
                try:
                    response = requests.post(f"{matcher_url}/match", json=message_for_Matcher) #, timeout=60)
                    response.raise_for_status()
                except requests.exceptions.RequestException as e:
                    print(f"Failed to connect to the remote API at {matcher_url}. Is it running? Error: {e}")
                    error_message = f"Failed to connect to the remote API at {matcher_url}. Is it running? Error: {e}"
                    return error_message, None, None, "error"
                    # Parse the JSON response from the server
                matcher_result = response.json()
                print(f"--- Real response from Ollama via remote API: {matcher_result} ---")
            
                matcher_version = matcher_result.get('matcher_version', 'UnknownMatcher')

                # matcher could not find any closely related cell_types
                # NOTE: adding more error checks and using .get(), which will return NoneType if missing, which is seemingly safer for type errors
                if not matcher_result or not matcher_result.get('cell_type_actual') or matcher_result.get('cell_type_actual') == MATCHER_NULL_RESPONSE:
                    print("No similar cell types were found using Matcher")
                    return request_error_msg, None, None, matcher_version
                else:
                    matched_cell_type = matcher_result['cell_type_actual']
                    print(f"Matcher cell type will now be used for accessibility: {matched_cell_type}")
                    
                    accessibility_tracks = simplified_targets_df[
                        (simplified_targets_df['Assay'].isin(['ATAC', 'DNASE'])) &
                        (simplified_targets_df['Cell Type'].str.lower() == matched_cell_type.lower())
                    ]
                    # print(accessibility_tracks)
                    return (
                        accessibility_tracks, matched_cell_type, (accessibility_tracks['Assay'].unique().tolist()), matcher_version
                        ) if not accessibility_tracks.empty else (request_error_msg, None, None, matcher_version)
        
        # 2. Expression (RNA for all request_type, except "expression_pol2")
        elif request_type in ["expression", "expression_pol2", "expression_mrna"]:


            cage_tracks = simplified_targets_df[
                (simplified_targets_df['Assay'] == 'CAGE') &
                # (simplified_targets_df['Cell Type'].str.lower() == cell_type)
                (simplified_targets_df['Cell Type'].str.lower() == cell_type)
            ]
            
            if not cage_tracks.empty:
                return cage_tracks, cell_type, ["CAGE"], "N/A"
            
            else:
                # Send request to Matcher
                print(f"No exact matching cell types in CAGE assay for cell type: {cell_type}. Querying Matcher for similar cell types in CAGE tracks.")
                #Send request to Matcher here
                cage_tracks_all = simplified_targets_df[(simplified_targets_df['Assay'] == 'CAGE')]
                #set up any dictionary to send to matcher
                message_for_Matcher = {
                    'cell_type_requested': cell_type,
                    'cell_type_list': cage_tracks_all['Cell Type'].unique().tolist()
                    }
                

                try:
                    response = requests.post(f"{matcher_url}/match", json=message_for_Matcher) #, timeout=60)
                    response.raise_for_status()
                except requests.exceptions.RequestException as e:
                    print(f"Failed to connect to the remote API at {matcher_url}. Is it running? Error: {e}")
                    error_message = f"Failed to connect to the remote API at {matcher_url}. Is it running? Error: {e}"
                    return error_message, None, None, "error"
                    # Parse the JSON response from the server
                matcher_result = response.json()
                print(f"--- Real response from Ollama via remote API: {matcher_result} ---")
            
                matcher_version = matcher_result.get('matcher_version', 'UnknownMatcher')

                #matcher could not find any closely related cell_types
                if not matcher_result or not matcher_result.get('cell_type_actual') or matcher_result.get('cell_type_actual') == MATCHER_NULL_RESPONSE:
                    print("No similar cell types were found using Matcher")
                    return request_error_msg, None, None, matcher_version
                else:
                    matched_cell_type = matcher_result['cell_type_actual']
                    print(f"Matcher cell type will now be used for CAGE: {matched_cell_type}")
                    
                    cage_tracks = simplified_targets_df[
                        (simplified_targets_df['Assay'] == 'CAGE') &
                        (simplified_targets_df['Cell Type'].str.lower() == matched_cell_type.lower())
                    ]
                
                    return (cage_tracks, matched_cell_type, ["CAGE"], matcher_version) if not cage_tracks.empty else (request_error_msg, None, None, matcher_version)
        
        # 3. Binding -- binding_{molecule}
        # (Parse CHIP assays, filter out TF binding/ histone modification molecule and cell_type)
        elif request_type.startswith("binding_"):
            chip_tracks = simplified_targets_df[
                (simplified_targets_df['Assay'] == 'CHIP') &
                (simplified_targets_df['Molecule'].str.lower() == molecule) &
                (simplified_targets_df['Cell Type'].str.lower() == cell_type)
            ]

            if not chip_tracks.empty:
                return chip_tracks, cell_type, [f"CHIP_{molecule}"], "N/A"

            else:
                # If no exact match, try to match the molecule first
                print(f"No exact matching cell type and molecule pairs in CHIP assay: {cell_type}, {molecule}. Querying Matcher for similar molecules.")
                #Send request to Matcher here
                chip_tracks_all = simplified_targets_df[
                    (simplified_targets_df['Assay'] == 'CHIP')
                    ]
                #set up any dictionary to send to matcher
                message_for_Matcher = {
                    'binding_molecule_requested': molecule,
                    'binding_molecule_list': chip_tracks_all['Molecule'].unique().tolist()
                    }
                
                try:
                    response = requests.post(f"{matcher_url}/match", json=message_for_Matcher) #, timeout=60)
                    response.raise_for_status()
                except requests.exceptions.RequestException as e:
                    print(f"Failed to connect to the remote API at {matcher_url}. Is it running? Error: {e}")
                    error_message = f"Failed to connect to the remote API at {matcher_url}. Is it running? Error: {e}"
                    return error_message, None, None, "error"
                    # Parse the JSON response from the server
                matcher_result = response.json()
                print(f"--- Real response from Ollama via remote API: {matcher_result} ---")
            
                matcher_version = matcher_result.get('matcher_version', 'UnknownMatcher')
                
                # matcher could not find any closely related cell_types
                if not matcher_result or not matcher_result.get('binding_molecule_actual') or matcher_result.get('binding_molecule_actual') == MATCHER_NULL_RESPONSE:
                    print("No similar molecule tracks were found using Matcher.")
                    return request_error_msg, None, None, matcher_version
                
                else:
                    # Got a matched molecule to proceed with
                    matched_molecule = matcher_result['binding_molecule_actual']
                    print(f"Matcher molecule will now be used for CHIP: {matched_molecule}")
                    
                    # With the matched molecule, try an exact match on the ORIGINAL cell type again.
                    chip_tracks = simplified_targets_df[
                        (simplified_targets_df['Assay'] == 'CHIP') &
                        (simplified_targets_df['Molecule'].str.lower() == matched_molecule.lower()) &
                        (simplified_targets_df['Cell Type'].str.lower() == cell_type)
                    ]   
                    
                    # If empty request Matcher to map cell_type from the newly filtered tracks
                    if not chip_tracks.empty:
                        return chip_tracks, cell_type, [f"CHIP_{matched_molecule}"], matcher_version
                    else:
                        # If that still fails, then call the matcher to map cell-type
                        # Send request to Matcher to map cell type next
                        print(f"No exact matching cell types in CHIP assay for Matcher-mapped molecule: {matched_molecule}. Querying Matcher for similar cell type in CHIP tracks.")            
                        #Send request to Matcher here
                        chip_tracks_molecule_mapped = simplified_targets_df[
                            (simplified_targets_df['Assay'] == 'CHIP') &
                            (simplified_targets_df['Molecule'].str.lower() == matched_molecule.lower())
                        ]
                        #set up any dictionary to send to matcher
                        message_for_Matcher = {
                            'cell_type_requested': cell_type,
                            'cell_type_list': chip_tracks_molecule_mapped['Cell Type'].unique().tolist()
                            }
                        
                        try:
                            response = requests.post(f"{matcher_url}/match", json=message_for_Matcher) #, timeout=60)
                            response.raise_for_status()
                        except requests.exceptions.RequestException as e:
                            print(f"Failed to connect to the remote API at {matcher_url}. Is it running? Error: {e}")
                            error_message = f"Failed to connect to the remote API at {matcher_url}. Is it running? Error: {e}"
                            return error_message, None, None, "error"
                            # Parse the JSON response from the server
                        
                        matcher_result = response.json()
                        print(f"--- Real response from Ollama via remote API: {matcher_result} ---")
                    
                        matcher_version = matcher_result.get('matcher_version', 'UnknownMatcher')
                        # Use the robust check for the cell type result
                        if not matcher_result or not matcher_result.get('cell_type_actual') or matcher_result.get('cell_type_actual') == MATCHER_NULL_RESPONSE:
                            print("No similar cell types were found using Matcher for the specified molecule.")
                            return request_error_msg, None, None, matcher_version
                        
                        else:
                            matched_cell_type = matcher_result['cell_type_actual']
                            print(f"Matcher cell type will now be used for CHIP_{matched_molecule}: {matched_cell_type}")

                            chip_tracks_cell_type_mapped = simplified_targets_df[
                                (simplified_targets_df['Assay'] == 'CHIP') &
                                (simplified_targets_df['Molecule'].str.lower() == matched_molecule.lower()) &
                                (simplified_targets_df['Cell Type'].str.lower() == matched_cell_type.lower())
                            ]
                            return (chip_tracks_cell_type_mapped, matched_cell_type, [f"CHIP_{matched_molecule}"], matcher_version) if not chip_tracks_cell_type_mapped.empty else (request_error_msg, None, None, matcher_version)
                
        # Invalid request type
        else:
            raise ValueError(f"Invalid request type {request_type}")

    except ConnectionError as e:
        print(f"A fatal error occurred while communicating with the Matcher: {e}")
        error_message = f"Internal Server Error: The dependent Matcher service at {matcher_ip}:{matcher_port} is unavailable."
        # Return a 4-element tuple to match the success signature and avoid crashing the caller
        return error_message, None, None, "error"
    