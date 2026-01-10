# enformer_predict_codebase.py
# August 12th, 2025
import os
import sys
import tqdm
import numpy as np
import pandas as pd
from collections import defaultdict

ENFORMER_SCRIPT_DIR = os.path.dirname(__file__)
ENFORMER_DIR = os.path.dirname(ENFORMER_SCRIPT_DIR)

sys.path.append(ENFORMER_DIR)
from Modules.Enformer import *
from Modules.FastaExt import *

from enformer_utils import *
from api_preprocessing_utils import *

targets_file = f"{ENFORMER_SCRIPT_DIR}/simplify_targets/enformer_targets_human.txt"
simplified_targets_file = f"{ENFORMER_SCRIPT_DIR}/simplify_targets/enformer_human_targets_simplified.txt"
# Simplified targets file was created to easily map the requested type and cell type
# to the right tracks. The python script for that is in `simplify_targets/` directory.

model = Enformer(f"{ENFORMER_SCRIPT_DIR}/trained_model/")

# Sequence parameters
SEQ_LEN = 393216
prediction_window = 114688
BIN_SIZE = 128

# 2. Load target files
def load_targets():
    targets_df = pd.read_csv(targets_file, index_col=0, sep='\t')
    simplified_targets_df = pd.read_csv(simplified_targets_file, index_col=0, sep='\t')
    return targets_df, simplified_targets_df

# 5. Prediction Function -- Runs Once and Filters Predictions Based on Request Type
def predict_enformer(sequences, request_tasks, matcher_ip, matcher_port, is_point_readout=False):
    """
    Runs the Enformer model on provided sequences and filters track predictions.

    This function processes a batch of sequences, determines the necessary model
    output tracks based on the requested tasks, calls a matcher service if needed,
    runs the model prediction once, and formats the output.
    
    Args:
        sequences (dict): A dictionary of key-value pairs {sequence_id: sequence}.
        request_tasks (set): A set of strings (request_type, cell_type) pairs 
                             to determine required tracks.
        matcher_ip (str): The IP address of the Matcher server.
        matcher_port (int): The port number of the Matcher server.
        is_point_readout (bool): If True, aggregates track predictions to a single
                                 value.
    
    Returns:
        tuple: A tuple containing:
            - task_predictions (dict or str): On success, a dictionary of
              prediction results. On failure, an error message string.
            - overall_matcher_version (str): The version of the matcher
              service used, or "N/A" if it was not called.
            
            For "all_tracks" tasks, predictions are not averaged over tracks
            and the full prediction matrix is returned (shape [1, num_sliced_bins, 5313 tracks]).
    """
    print("Running Enformer Model Predictions on ALL tracks!")
    
    # 5.1. Collect all required track indices
    print("Collecting track indices for required tasks...")
    task_to_indices = {} # Dictionary to store required track indices from each task
    # Example: {('expression', 'H1'): [1, 3],
    #           ('accessibility', 'K562'): [2],
    #           ('expression_pol2', 'H1'): [1, 3]} (fallback to RNA:H1,
    #                                               since there are no CAGE:H1 tracks)

    track_to_tasks = defaultdict(set) # Maps each track index to a set of tasks that require it.
                                      # This prevents predicting on the same track twice.
    # Example: {1: {('expression', 'H1'), ('epression_pol2', 'H1')},  # Track 1 needed by both expression tasks
    #           3: {('expression', 'H1'), ('expression_pol2', 'H1')},  # Track 3 needed by both expression tasks
    #           2: {('accessibility', 'K562')}}  # Track 2 needed only by accessibility task

    unique_track_indices = set() # Stores all the unique tracks needed for prediction
                                 # ensuring we only process relevant tracks once!
    # Example: [1, 2, 3]
    
    # Initialize variable to hold Matcher version
    overall_matcher_version = "N/A"
    
    # NEW: Load targets ONCE before loop
    targets_df, simplified_targets_df = load_targets()
    
    for request_type, cell_type in request_tasks:
        print(f"Performing track selection for {request_type} and {cell_type}")
        # Get track indices of desired tracks for filtering predictions
        filtered_tracks, cell_type_actual, type_actual, task_matcher_version = filter_evaluator_request(simplified_targets_df,
                                                request_type, cell_type, matcher_ip, matcher_port)
        
        if task_matcher_version not in ["N/A", "error"]:
            overall_matcher_version = task_matcher_version
            
        task_key = (request_type, cell_type)
        # If filtered_tracks returns a string, it is an error message -- bail-out!
        if isinstance(filtered_tracks, str):
            # The function failed, and 'filtered_tracks' now holds the error message.
            # Return this error message to the API to be sent to the client.
            #return filtered_tracks, overall_matcher_version
            print(f"No matching tracks found for {request_type} and {cell_type}. Skipping...")
            task_to_indices[task_key] = {"error": filtered_tracks}
        else:
            # Otherwise, proceed as before -- knowing filtered_tracks is a DataFrame
            track_indices = filtered_tracks.index.tolist()
            if not track_indices:
                print(f"An error occured during pulling the tracks.")
                continue
            # Avoid printing a huge list for "all_tracks" requests
            # NOTE: Setting it up so that Enformer cannot process all_tracks request
            # NOTE: The error checking will catch it and will not let it get to this stage. I just did not want to change the code a whole lot.
            if request_type.lower() == "all_tracks":
                print(f"Enformer does not handle the 'all_tracks' request_type.")
                continue
            else:
                print(f"Using Track Indices for ({request_type}, {cell_type}): {track_indices}")
        
            task_to_indices[task_key] = {
                "track_indices": track_indices,
                "cell_type_actual": cell_type_actual,
                "type_actual": type_actual
            }
            for index in track_indices:
                track_to_tasks[index].add(task_key) # Mapping track to tasks
                
            unique_track_indices.update(track_indices)
    # Convert to sorted list to maintain order -- easy to test
    unique_track_indices = sorted(list(unique_track_indices))
    
    # Check if any request is an all_tracks request 
    # NOTE: The error checking will catch it and will not let it get to this stage. I just did not want to change the code a whole lot. 
    if any(rt.lower() == "all_tracks" for rt, _ in request_tasks):
        print(f"Enformer does not handle the 'all_tracks' request_type.")
        print(error_msg)
        return error_msg, overall_matcher_version
    else:
        print(f"Unique required track indices for all tasks: {unique_track_indices}")
    
    task_predictions = task_to_indices
    #If no tracks were found for any of the requested tasks return the meta data with the errors stored in the predictions
    #No predictions need to be made
    if not unique_track_indices:
        error_msg = "No valid track indices found for any tasks."
        print(error_msg)
        for task_key, values in task_to_indices.items():
                task_predictions[task_key] = values
        return task_predictions, overall_matcher_version

    # 5.2. Process each sequence and run prediction
    # Iterate over sequences and run model prediction only for the required tracks
    print("Processing sequences and predicting only on required tracks...")

    # Process each sequence
    for seq_id, sequence in tqdm.tqdm(sequences.items(),
                                      desc="Predictions in progress",
                                      unit="sequence",
                                      total=len(sequences),
                                      dynamic_ncols=True):

        print(f"The length of the current sequence is: {len(sequence)}")

        #CASE 1: Sequences <= 114kb: center then on the receptive field and predict, crop N bins from both side
        if len(sequence) <= prediction_window:
            print("For sequences shorter than the prediction window only make one prediction")
            # Pad and encode sequence to 393kb
            encoded_seq = pad_sequence(sequence, SEQ_LEN)
            encoded_seq = one_hot_encode(encoded_seq)

            # Run model prediction once for all required tracks
            predictions = model.predict_on_batch(encoded_seq[np.newaxis])['human'][0][:, unique_track_indices]
            
            #This slicing function works for sequences centered on the prediction window
            sliced_predictions, trim_upstream = slice_prediction_tracks(
                full_track=predictions[np.newaxis, :, :],
                original_seq_len=len(sequence),
                model_input_len=SEQ_LEN//2,
                bin_size=BIN_SIZE
            )

            sliced_predictions = sliced_predictions.squeeze(0)

        #CASE 2: Sequences >114kb
        #Declare list to append predictions
        predictions = []
        if len(sequence) > prediction_window:
            #If you sequence is longer than the 114k prediction window you have to make multiple predictions
            #Pad upstream of the sequence so that the first prediction from the model corresponds to the first base on the sequence
            sequence_with_upstream_pad = ('N' * (40960)) + sequence
            print(f"Length of the sequence with upstream Ns added is: + {len(sequence_with_upstream_pad)}")

            #Marks how much of the actual sequence has been predicted on
            seq_predicted_end = 40960
            start_pos = 0

            #Start predictions
            while seq_predicted_end < len(sequence_with_upstream_pad):
                #The end position is either min start position of the current sequence or the end of the sequence
                end_pos = min(len(sequence_with_upstream_pad), start_pos+196608)
                print("Start prediction loop")

                #Current sequence chunk
                seq_chunk = sequence_with_upstream_pad[start_pos:end_pos]

                #If there is enough sequence to fit into the Enformer window don't need to add extra padding
                #No need to crop the prediction bins either
                if len(seq_chunk) == 196608:
                    print("For sequence chunks that match the model's input length")
                    #Pad up to enformer's 393kb
                    encoded_seq = pad_sequence(seq_chunk, SEQ_LEN)

                    encoded_seq = one_hot_encode(encoded_seq)
                    pred_chunk =  model.predict_on_batch(encoded_seq[np.newaxis])['human'][0][:, unique_track_indices]
             
                    #Append prediction to running sequence prediction
                    predictions.append(pred_chunk)

                    #114kb of the sequence was predicted, move the counters forward
                    seq_predicted_end = seq_predicted_end + 114688 
                    start_pos = start_pos + 114688

                else:
                    #If the amount of sequence you can pull is less than 196kb we have two possible options
                    print("For sequence chunks shorter than the model's input length")
                    #CASE 2A: If the sequence doesn't fit into the 114kb prediction window, you will need to make 2 predictions to predict for each base
                    if len(seq_chunk) > 114688:
                        #Pad only downstream so sequence is not centered, but the first base is at the start of the first bin
                        downstream_pad = 196608 - len(seq_chunk)
                        seq_chunk_downstreamN = seq_chunk + ('N' * (downstream_pad))

                        encoded_seq = pad_sequence(seq_chunk_downstreamN, SEQ_LEN)
                        encoded_seq = one_hot_encode(encoded_seq)
                        pred_chunk =  model.predict_on_batch(encoded_seq[np.newaxis])['human'][0][:, unique_track_indices]
                        predictions.append(pred_chunk)
                        
                        #Next 114kb was predicted
                        #Increment both trackers
                        start_pos = start_pos + 114688
                        seq_predicted_end = seq_predicted_end + 114688

                    #if what's left fits into one prediction window we only need to make one prediction
                    #Only end up here at the very end of the sequence
                    elif len(seq_chunk) <= 114688:
                        #Calculate how much we need to pad downstream
                        downstream_pad = 196608 - len(seq_chunk)
                        seq_chunk_downstreamN = seq_chunk + ('N' * (downstream_pad))

                        encoded_seq = pad_sequence(seq_chunk_downstreamN, SEQ_LEN)
                        encoded_seq = one_hot_encode(encoded_seq)
                        pred_chunk =  model.predict_on_batch(encoded_seq[np.newaxis])['human'][0][:, unique_track_indices]
                        
                        #How many bases were extra in the enformer prediction window that we need to remove
                        bases_to_crop_from_end = downstream_pad - 40960
                        #Calculate the number of bins to crop, round down so you don't remove any bins with "real" sequence
                        bins_to_crop_from_end = bases_to_crop_from_end//128

                        if bins_to_crop_from_end == 0:
                            #If it's exactly 114kb keep all bins
                            predictions.append(pred_chunk)
                        else:
                            #Crop the full N bins downstream
                            predictions.append(pred_chunk[:-bins_to_crop_from_end, :])

                        #Sanity check how many bases would we need to trip downstream
                        trim_downstream = bases_to_crop_from_end - (bins_to_crop_from_end*128) 
                        #Since trim_downstream is implied, no need to return
                        #For sequences longer than the input window the extra base trimming will only happen from the downstream bin
                        trim_upstream = 0
                        break

            sliced_predictions = np.concatenate(predictions, axis=0)
            print("Final predictions shape:", sliced_predictions.shape)

        # Now assign filtered predictions to each task to be averaged
        for task_key, values in task_to_indices.items():

            if "error" in values:
                task_predictions[task_key] = values
            else:
                # Extract relevant track predictions per task
                selected_tracks = sliced_predictions[:, [unique_track_indices.index(idx) for idx in values['track_indices']]]
                # Average duplicate tracks per task
                avg_prediction = np.mean(selected_tracks, axis=-1, keepdims=True)
                
                if is_point_readout:
                    # "point" readout: Average across 896 bins to a single value per sequence
                    point_prediction = np.mean(avg_prediction, axis=0) # =1, keepdims=True)
                    task_predictions[task_key][seq_id] = point_prediction.squeeze().tolist()
                else:
                    # "track" readout: Return full 896 bin predictions
                    # Store predictions in task-specific dictionary
                    task_predictions[task_key][seq_id] = avg_prediction.squeeze().tolist()
                    #Need to add trim upstream if it exists
                    task_predictions[task_key].setdefault("trim_upstream", {})[seq_id] = trim_upstream
                    
    return task_predictions, overall_matcher_version


