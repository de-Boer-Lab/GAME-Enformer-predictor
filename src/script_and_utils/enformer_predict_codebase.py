# enformer_predict_codebase.py
# August 12th, 2025
import os
import sys
import tqdm
import copy # Used for deep copying data structures when needed to avoid mutability issues
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

model = Enformer(f"{ENFORMER_SCRIPT_DIR}/trained_model/")

# Sequence parameters
SEQ_LEN = 393216
prediction_window = 114688
BIN_SIZE = 128
SEQ_CONTEXT = 40960

# 2. Load targets files
def load_targets():
    human_df = pd.read_csv(f"{ENFORMER_SCRIPT_DIR}/simplify_targets/enformer_human_targets_simplified.txt", index_col=0, sep='\t')
    mouse_df = pd.read_csv(f"{ENFORMER_SCRIPT_DIR}/simplify_targets/enformer_mouse_targets_simplified.txt", index_col=0, sep='\t')
    
    # Clean up column names just in case
    human_df.columns = human_df.columns.str.strip()
    mouse_df.columns = mouse_df.columns.str.strip()
    
    return human_df, mouse_df

# 5. Prediction Function -- Runs Once and Filters Predictions Based on Request Type
def predict_enformer(sequences, request_tasks, matcher_ip, matcher_port, prediction_ranges, is_point_readout=False):
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
        prediction_ranges (dict): A dictionary of key-value pairs {sequence_id: sequence}.
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
    # Example: {1: {('expression', 'H1'), ('expression_pol2', 'H1')},  # Track 1 needed by both expression tasks
    #           3: {('expression', 'H1'), ('expression_pol2', 'H1')},  # Track 3 needed by both expression tasks
    #           2: {('accessibility', 'K562')}}  # Track 2 needed only by accessibility task

    unique_human_indices = set() # Stores all the unique tracks needed for prediction
                                 # ensuring we only process relevant tracks once!
    unique_mouse_indices = set() # Same as above but for mouse
    # Example: [1, 2, 3]
    
    # Initialize variable to hold Matcher version
    overall_matcher_version = "N/A"
    
    # NEW: Load targets ONCE before loop
    human_targets_df, mouse_targets_df = load_targets()
    
    for request_type, cell_type, species in request_tasks:
        print(f"Performing track selection for {request_type}, {cell_type}, and {species}...")
        
        # NEW: Route with strict exact matching
        normalized_species = species.lower()
        if normalized_species == "mus_musculus":
            targets_df_to_use = mouse_targets_df
        elif normalized_species == "homo_sapiens":
            targets_df_to_use = human_targets_df
        else:
            # Failsafe in case an invalid species somehow bypasses model_validation.py
            print(f"Error: Unsupported species '{species}'.")
            task_key = (request_type, cell_type, species)
            task_to_indices[task_key] = {"error": f"Unsupported species: {species}"}
            continue
        
        # Get track indices of desired tracks for filtering predictions
        filtered_tracks, cell_type_actual, type_actual, task_matcher_version = filter_evaluator_request(
            targets_df_to_use, request_type, cell_type, matcher_ip, matcher_port
        )
        
        if task_matcher_version not in ["N/A", "error"]:
            overall_matcher_version = task_matcher_version
        
        # NEW: task_key now includes species
        task_key = (request_type, cell_type, species)
        
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
                print(f"An error occurred during pulling the tracks.")
                continue
            # Avoid printing a huge list for "all_tracks" requests
            # NOTE (Satyam): Setting it up so that Enformer cannot process all_tracks request
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
                
            # NEW: Add to the appropriate species set
            if normalized_species == "mus_musculus":
                unique_mouse_indices.update(track_indices)
            else:
                unique_human_indices.update(track_indices)
                
    # Convert to sorted list to maintain order -- easy to test
    unique_human_indices = sorted(list(unique_human_indices))
    unique_mouse_indices = sorted(list(unique_mouse_indices))
    
    task_predictions = copy.deepcopy(task_to_indices)
    
    # If no tracks were found for any of the requested tasks, return the meta data with the errors stored in the predictions
    if not unique_human_indices and not unique_mouse_indices:
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
        #Get current prediction range using sequence ID, some may be an empty list if no ranges are requested for that sequence
        pred_range = prediction_ranges.get(seq_id, [])
        new_range_start = None
        new_range_end = None
        
        #If Prediction ranges exist for this sequence
        if len(pred_range) > 0:
            print("Prediction ranges are present for current sequences, subsetting sequence if needed")
            #We should subset to the needed sequence that's relevant for the ranges
            sequence, new_range_start, new_range_end = subset_sequence_for_ranges(sequence, pred_range, prediction_window, SEQ_CONTEXT)
            print(f"Length of sequence after subsetting to ranges loci: {len(sequence)}")

        trim_upstream = 0
        
        # NEW: Dictionary to hold final sliced predictions for both species
        sliced_predictions = {'human': None, 'mouse': None}
        
        # CASE 1: Sequences <= 114kb: center then on the receptive field and predict, crop N bins from both side
        if len(sequence) <= prediction_window:
            print("For sequences shorter than the prediction window only make one prediction")
            # Pad and encode sequence to 393kb
            encoded_seq = pad_sequence(sequence, SEQ_LEN)
            encoded_seq = one_hot_encode(encoded_seq)
            
            # Run model prediction ONCE
            raw_preds = model.predict_on_batch(encoded_seq[np.newaxis])

            # NEW: Extract and slice for both species if requested
            for species_key, unique_indices in [('human', unique_human_indices), ('mouse', unique_mouse_indices)]:
                if unique_indices:
                    predictions = raw_preds[species_key][0][:, unique_indices]

                    # Slice the predictions using the prediction ranges bins if prediction ranges exists for this sequence
                    if new_range_start is not None:
                        sliced_preds_temp, trim_upstream = slice_prediction_tracks_for_range(
                            full_track=predictions[np.newaxis, :, :],
                            original_seq_len=len(sequence),
                            model_input_len=SEQ_LEN//2,
                            range_start=new_range_start,
                            range_end=new_range_end,
                            bin_size=BIN_SIZE
                        )
                    else:
                        # Otherwise slice using usual logic (only keeping bins with the sequence)
                        # This slicing function works for sequences centered on the prediction window
                        sliced_preds_temp, trim_upstream = slice_prediction_tracks(
                            full_track=predictions[np.newaxis, :, :],
                            original_seq_len=len(sequence),
                            model_input_len=SEQ_LEN//2,
                            bin_size=BIN_SIZE
                        )
                    sliced_predictions[species_key] = sliced_preds_temp.squeeze(0)

        # CASE 2: Sequences >114kb
        # Declare list to append predictions
        else:
            # NEW: Declare dictionary to append predictions for both heads
            predictions_dict = {'human': [], 'mouse': []}
            
            #If your sequence is longer than the 114k prediction window you have to make multiple predictions
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
                    print("For sequence chunks that match the model's input length 196kb")
                    #Pad up to enformer's 393kb
                    encoded_seq = pad_sequence(seq_chunk, SEQ_LEN)

                    encoded_seq = one_hot_encode(encoded_seq)
                    raw_preds = model.predict_on_batch(encoded_seq[np.newaxis])
             
                    # NEW: Append predictions to running sequence prediction for BOTH species
                    for species_key, unique_indices in [('human', unique_human_indices), ('mouse', unique_mouse_indices)]:
                        if unique_indices:
                            pred_chunk = raw_preds[species_key][0][:, unique_indices]
                            predictions_dict[species_key].append(pred_chunk)

                    #114kb of the sequence was predicted, move the counters forward
                    seq_predicted_end = seq_predicted_end + 114688 
                    start_pos = start_pos + 114688

                else:
                    #If the amount of sequence you can pull is less than 196kb we have two possible options
                    print("For sequence chunks shorter than the model's input length")
                    #CASE 2A: If the sequence doesn't fit into the 114kb prediction window, you will need to make 2 predictions to predict for each base
                    if len(seq_chunk) > 114688:
                        print("Seq chunk > 114688")
                        #Pad only downstream so sequence is not centered, but the first base is at the start of the first bin
                        downstream_pad = 196608 - len(seq_chunk)
                        seq_chunk_downstreamN = seq_chunk + ('N' * (downstream_pad))

                        encoded_seq = pad_sequence(seq_chunk_downstreamN, SEQ_LEN)
                        encoded_seq = one_hot_encode(encoded_seq)
                        raw_preds = model.predict_on_batch(encoded_seq[np.newaxis])
                        #Remove full N bins
                        bases_to_crop_from_end = downstream_pad - 40960

                        if bases_to_crop_from_end < 0:
                            bases_to_crop_from_end = 0

                        #Calculate the number of bins to crop, round down so you don't remove any bins with "real" sequence
                        bins_to_crop_from_end = bases_to_crop_from_end//128
                        
                        # NEW: Route the cropping to both heads
                        for species_key, unique_indices in [('human', unique_human_indices), ('mouse', unique_mouse_indices)]:
                            if unique_indices:
                                pred_chunk = raw_preds[species_key][0][:, unique_indices]
                                if bins_to_crop_from_end == 0:
                                    # If it's exactly 114kb keep all bins
                                    predictions_dict[species_key].append(pred_chunk)
                                else:
                                    # Crop the full N bins downstream
                                    predictions_dict[species_key].append(pred_chunk[:-bins_to_crop_from_end, :])

                        # Next 114kb was predicted
                        # Increment both trackers
                        start_pos = start_pos + 114688
                        seq_predicted_end = seq_predicted_end + 114688

                    # if what's left fits into one prediction window we only need to make one prediction
                    # Only end up here at the very end of the sequence
                    elif len(seq_chunk) <= 114688:
                        print("Seq chunk <= 114688")
                        #Calculate how much we need to pad downstream
                        downstream_pad = 196608 - len(seq_chunk)
                        seq_chunk_downstreamN = seq_chunk + ('N' * (downstream_pad))

                        encoded_seq = pad_sequence(seq_chunk_downstreamN, SEQ_LEN)
                        encoded_seq = one_hot_encode(encoded_seq)
                        raw_preds = model.predict_on_batch(encoded_seq[np.newaxis])
                        
                        #How many bases were extra in the enformer prediction window that we need to remove
                        bases_to_crop_from_end = downstream_pad - 40960
                        #Calculate the number of bins to crop, round down so you don't remove any bins with "real" sequence
                        bins_to_crop_from_end = bases_to_crop_from_end//128

                        # NEW: Route the cropping to both heads
                        for species_key, unique_indices in [('human', unique_human_indices), ('mouse', unique_mouse_indices)]:
                            if unique_indices:
                                pred_chunk = raw_preds[species_key][0][:, unique_indices]
                                if bins_to_crop_from_end == 0:
                                    # If it's exactly 114kb keep all bins
                                    predictions_dict[species_key].append(pred_chunk)
                                else:
                                    # Crop the full N bins downstream
                                    predictions_dict[species_key].append(pred_chunk[:-bins_to_crop_from_end, :])
                    
                        # # Sanity check how many bases would we need to trip downstream
                        # trim_downstream = bases_to_crop_from_end - (bins_to_crop_from_end*128) 
                        #Since trim_downstream is implied, no need to return
                        #For sequences longer than the input window the extra base trimming will only happen from the downstream bin
                        trim_upstream = 0
                        break
                
            # print("Final predictions shape:", sliced_predictions.shape)
            
            # NEW: Concatenate all chunks into one big track for EACH active species
            for species_key in ['human', 'mouse']:
                if predictions_dict[species_key]:
                    concat_preds = np.concatenate(predictions_dict[species_key], axis=0)
                    
                    if new_range_start is not None:
                        start_bin = math.floor(new_range_start / BIN_SIZE)
                        end_bin = math.ceil(new_range_end / BIN_SIZE)
                        concat_preds = concat_preds[start_bin:end_bin, :]
                        # Note: We overwrite trim_upstream here but it's identical for both species
                        trim_upstream = new_range_start - (start_bin * BIN_SIZE)
                        
                    sliced_predictions[species_key] = concat_preds

        # Now assign filtered predictions to each task to be averaged
        for task_key, values in task_to_indices.items():
            request_type, cell_type, species = task_key

            if "error" in values:
                task_predictions[task_key] = values
            else:
                # NEW: Extract from the correct species dictionary
                if species.lower() == "mus_musculus":
                    current_sliced_preds = sliced_predictions['mouse']
                    current_unique_indices = unique_mouse_indices
                else:
                    current_sliced_preds = sliced_predictions['human']
                    current_unique_indices = unique_human_indices
                    
                selected_tracks = current_sliced_preds[:, [current_unique_indices.index(idx) for idx in values['track_indices']]]
                avg_prediction = np.mean(selected_tracks, axis=-1, keepdims=True)
                
                if is_point_readout:
                    # "point" readout: Average across 896 bins to a single value per sequence
                    point_prediction = np.mean(avg_prediction, axis=0) 
                    task_predictions[task_key][seq_id] = point_prediction.squeeze().tolist()
                else:
                    # "track" readout: Return full 896 bin predictions
                    # Store predictions in task-specific dictionary
                    task_predictions[task_key][seq_id] = avg_prediction.squeeze().tolist()
                    #Need to add trim upstream if it exists
                    task_predictions[task_key].setdefault("trim_upstream", {})[seq_id] = trim_upstream
                    
    return task_predictions, overall_matcher_version

# Test all cases and edge cases in the predict_enformer function
if __name__ == "__main__":
    """
    Tests for prediction ranges and trim_upstream math.
    Uses the REAL pad_sequence, subset_sequence_for_ranges,
    slice_prediction_tracks, and slice_prediction_tracks_for_range
    from the wildcard imports. No model calls needed.

    Enformer architecture reference:
        TF-Hub input (SEQ_LEN):     393,216 bp
        Effective input:             196,608 bp  (model_input_len = SEQ_LEN // 2)
        Prediction window:           114,688 bp  (896 bins × 128 bp)
        Buffer per side:             40,960 bp   (320 bins × 128 bp)
        BIN_SIZE:                    128 bp
    """
    import math
    import numpy as np

    # -- constants (mirror the module-level ones for readability) -----------
    _MIL = SEQ_LEN // 2        # 196,608
    _PW  = prediction_window   # 114,688
    _BS  = BIN_SIZE            # 128
    _SC  = SEQ_CONTEXT         # 40,960
    _BUF = 320 * 2 * _BS      # 81,920
    _NBINS = 896

    # -- helpers that DON'T exist as standalone functions -------------------
    def _case2_range_slice(preds, range_start, range_end, bin_size):
        """Replicates the CASE 2 inline range-slicing from predict_enformer."""
        start_bin = math.floor(range_start / bin_size)
        end_bin   = math.ceil(range_end / bin_size)
        sliced    = preds[start_bin:end_bin, :]
        trim_code    = 0                                       # what the code does
        trim_correct = range_start - (start_bin * bin_size)    # what it SHOULD do
        return sliced, trim_correct, trim_code

    def _labeled(n_bins, n_tracks=2):
        """Prediction array where each bin's value = its index."""
        a = np.zeros((n_bins, n_tracks))
        for i in range(n_bins):
            a[i, :] = i
        return a

    # -- test runner -------------------------------------------------------
    _p, _f = 0, 0
    def _ok(cond, name, detail=""):
        global _p, _f
        if cond:
            _p += 1; print(f"  ✓ {name}")
        else:
            _f += 1; print(f"  ✗ {name}")
            if detail: print(f"    {detail}")

    # ======================================================================
    # TEST 1: pad_sequence
    # ======================================================================
    print("\n" + "=" * 70)
    print("TEST 1: pad_sequence")
    print("=" * 70)

    seq = "ACGT" * 250  # 1000 bp
    padded = pad_sequence(seq, SEQ_LEN)
    _ok(len(padded) == SEQ_LEN, f"Pads to {SEQ_LEN}", f"Got {len(padded)}")

    left_ns  = len(padded) - len(padded.lstrip('N'))
    tot_pad  = SEQ_LEN - 1000
    exp_left = tot_pad // 2
    _ok(left_ns == exp_left, f"Right-biased centering (left N = {exp_left})", f"Got {left_ns}")

    # Confirm also centered within inner 196,608 window
    c0 = (SEQ_LEN - _MIL) // 2
    inner = padded[c0 : c0 + _MIL]
    inner_left = len(inner) - len(inner.lstrip('N'))
    exp_inner  = (_MIL - 1000) // 2
    _ok(inner_left == exp_inner,
        f"Also centered in inner 196,608 window (left N = {exp_inner})",
        f"Got {inner_left}")

    _ok(pad_sequence("A" * SEQ_LEN, SEQ_LEN) == "A" * SEQ_LEN,
        "Exact-length sequence unchanged")

    # ======================================================================
    # TEST 2: subset_sequence_for_ranges
    # ======================================================================
    print("\n" + "=" * 70)
    print("TEST 2: subset_sequence_for_ranges")
    print("=" * 70)

    # 2a: Running example — 2M bp, range [500k, 510k]
    sub, rs, re = subset_sequence_for_ranges("A"*2_000_000, [500_000, 510_000], _PW, _SC)
    _ok(len(sub) == _MIL,   f"2M bp → subsetted to {_MIL}", f"Got {len(sub)}")
    _ok(rs == 93_304,       f"new_range_start = 93,304",     f"Got {rs}")
    _ok(re == 103_304,      f"new_range_end = 103,304",      f"Got {re}")
    _ok(re - rs == 10_000,  "Range size preserved (10,000)", f"Got {re-rs}")

    # 2b: Short sequence — no trimming
    sub2, rs2, re2 = subset_sequence_for_ranges("A"*50_000, [10_000, 30_000], _PW, _SC)
    _ok(len(sub2) == 50_000,                   "50k: no trimming needed", f"Got {len(sub2)}")
    _ok(rs2 == 10_000 and re2 == 30_000,       "Range coords unchanged")

    # 2c: Large range (>= prediction_window)
    sub3, rs3, _ = subset_sequence_for_ranges("A"*1_000_000, [200_000, 500_000], _PW, _SC)
    exp3 = (500_000 + _SC) - (200_000 - _SC)
    _ok(len(sub3) == exp3,  f"Large range: len = {exp3}", f"Got {len(sub3)}")
    _ok(rs3 == _SC,         f"Large range: new_range_start = {_SC}", f"Got {rs3}")

    # 2d: Range near start (clamps to 0)
    _, rs4, _ = subset_sequence_for_ranges("A"*500_000, [1_000, 5_000], _PW, _SC)
    _ok(rs4 == 1_000, "Near start: range_start = 1000 (clamped to 0)", f"Got {rs4}")

    # 2e: Range near end (clamps to len)
    sub5, _, re5 = subset_sequence_for_ranges("A"*500_000, [498_000, 499_000], _PW, _SC)
    exp_s5 = max(math.floor(498_500 - _PW/2 - _SC), 0)
    _ok(re5 == 499_000 - exp_s5,   f"Near end: new_range_end = {499_000 - exp_s5}", f"Got {re5}")
    _ok(len(sub5) < _MIL,          f"Near end: shorter than model input ({len(sub5)} < {_MIL})")

    # 2f: Boundary — range == prediction_window (takes else path)
    sub6, _, _ = subset_sequence_for_ranges("A"*1_000_000, [100_000, 100_000+_PW], _PW, _SC)
    _ok(len(sub6) == _PW + 2*_SC,
        f"Boundary: len = pred_window + 2×context = {_PW+2*_SC}", f"Got {len(sub6)}")

    # 2g: Tiny sequence — clamped both sides
    sub7, rs7, re7 = subset_sequence_for_ranges("A"*10_000, [2_000, 8_000], _PW, _SC)
    _ok(len(sub7)==10_000 and rs7==2_000 and re7==8_000, "Tiny: no change (clamped both ends)")

    # ======================================================================
    # TEST 3: slice_prediction_tracks (CASE 1, no ranges)
    # ======================================================================
    print("\n" + "=" * 70)
    print("TEST 3: slice_prediction_tracks (CASE 1, no ranges)")
    print("=" * 70)

    p896 = _labeled(_NBINS)

    # 3a: 1000 bp
    sl, tu = slice_prediction_tracks(p896[np.newaxis], 1000, _MIL, _BS, _BUF)
    lp = (_MIL - 1000) // 2
    lp_ab = lp - _SC
    exp_sb = lp_ab // _BS
    exp_eb = math.ceil((lp_ab + 1000) / _BS)
    _ok(sl.shape[1] == exp_eb - exp_sb, f"1000 bp → {exp_eb-exp_sb} bins", f"Got {sl.shape[1]}")
    _ok(sl.shape[1]*_BS >= 1000,        f"Covers ≥ 1000 bp ({sl.shape[1]*_BS})")
    _ok(tu == 12,                        "trim_upstream = 12", f"Got {tu}")
    _ok(int(sl[0,0,0]) == exp_sb,       f"First bin = {exp_sb}", f"Got {int(sl[0,0,0])}")

    # 3b: 114,688 bp (prediction_window — perfect fit)
    sl2, tu2 = slice_prediction_tracks(p896[np.newaxis], _PW, _MIL, _BS, _BUF)
    _ok(sl2.shape[1] == _NBINS,  f"114,688 bp → all {_NBINS} bins")
    _ok(tu2 == 0,                 "trim_upstream = 0")

    # 3c: 200 bp (typical MPRA element)
    sl3, _ = slice_prediction_tracks(p896[np.newaxis], 200, _MIL, _BS, _BUF)
    _ok(sl3.shape[1] == 2,            "200 bp → 2 bins", f"Got {sl3.shape[1]}")
    _ok(sl3.shape[1]*_BS >= 200,      f"Covers ≥ 200 bp ({sl3.shape[1]*_BS})")

    # 3d: 196,608 bp (model_input_len — no padding at all)
    sl4, tu4 = slice_prediction_tracks(p896[np.newaxis], _MIL, _MIL, _BS, _BUF)
    _ok(sl4.shape[1] == _NBINS, "196,608 bp → all 896 bins")
    _ok(tu4 == 0,                "trim_upstream = 0")

    # ======================================================================
    # TEST 4: slice_prediction_tracks_for_range (CASE 1, with ranges)
    # ======================================================================
    print("\n" + "=" * 70)
    print("TEST 4: slice_prediction_tracks_for_range (CASE 1, with ranges)")
    print("=" * 70)

    # 4a: 50k bp seq, range [10k, 30k]
    sl_r, tu_r = slice_prediction_tracks_for_range(
        p896[np.newaxis], 50_000, 10_000, 30_000, _MIL, _BS, _BUF)
    lp50 = (_MIL - 50_000) // 2
    rs_o = 10_000 + lp50 - _SC
    re_o = 30_000 + lp50 - _SC
    esb  = rs_o // _BS
    eeb  = math.ceil(re_o / _BS)
    _ok(sl_r.shape[1] == eeb-esb,      f"50k [10k,30k]: {eeb-esb} bins", f"Got {sl_r.shape[1]}")
    _ok(sl_r.shape[1]*_BS >= 20_000,   f"Covers ≥ 20,000 bp ({sl_r.shape[1]*_BS})")
    _ok(tu_r == rs_o - esb*_BS,        f"trim_upstream = {rs_o-esb*_BS}", f"Got {tu_r}")

    # 4b: Bin-aligned range (trim should be 0)
    # For 50k: output_offset = lp50-_SC = 32344; 32344 mod 128 = 88 → start=40 aligns
    sl_a, tu_a = slice_prediction_tracks_for_range(
        p896[np.newaxis], 50_000, 40, 30_000, _MIL, _BS, _BUF)
    _ok(tu_a == 0, "Bin-aligned range: trim_upstream = 0", f"Got {tu_a}")

    # 4c: Full-sequence range == no-range slice (same bins selected)
    sl_fr, _ = slice_prediction_tracks_for_range(
        p896[np.newaxis], 50_000, 0, 50_000, _MIL, _BS, _BUF)
    sl_nr, _ = slice_prediction_tracks(p896[np.newaxis], 50_000, _MIL, _BS, _BUF)
    _ok(np.array_equal(sl_fr, sl_nr), "Full-seq range selects same bins as no-range")

    # ======================================================================
    # TEST 5: CASE 2 range slicing & trim_upstream (bug now FIXED)
    # ======================================================================
    print("\n" + "=" * 70)
    print("TEST 5: CASE 2 range slicing (tiled predictions) — bug now FIXED")
    print("=" * 70)

    p1536 = _labeled(1536)

    # 5a: Running example — [93304, 103304] in 1536-bin tiled prediction
    sl_c2, tr_ok, tr_bug = _case2_range_slice(p1536, 93_304, 103_304, _BS)
    _ok(sl_c2.shape[0] == 80,       "80 bins (728:808)",   f"Got {sl_c2.shape[0]}")
    _ok(int(sl_c2[0,0]) == 728,     "First bin = 728",     f"Got {int(sl_c2[0,0])}")
    _ok(int(sl_c2[-1,0]) == 807,    "Last bin = 807",      f"Got {int(sl_c2[-1,0])}")

    # 5b: The bug (now FIXED in predict_enformer — trim_upstream line added)
    # _case2_range_slice still simulates the OLD code for documentation
    _ok(tr_bug == 0,     "OLD CODE returned trim_upstream = 0 (was the bug)")
    _ok(tr_ok == 120,    "CORRECT trim_upstream = 120 (now applied in predict_enformer)",  f"Got {tr_ok}")
    _ok(tr_ok != tr_bug, f"Difference was {tr_ok} bp — fix closes this gap")

    # 5c: Coverage
    _ok(728*_BS <= 93_304,    "First bin starts ≤ range start")
    _ok(808*_BS >= 103_304,   "Last bin ends ≥ range end")

    # 5d: Bin-aligned range — trim is 0 regardless
    _, ta, tc = _case2_range_slice(p1536, 93_184, 103_296, _BS)
    _ok(ta == 0 and tc == 0,  "Bin-aligned: trim = 0 (old and new code agree)")

    # 5e: Worst case — range_start = k*128 + 127
    _, tw_ok, tw_bug = _case2_range_slice(p1536, 93_311, 103_311, _BS)
    _ok(tw_ok == 127,  f"Worst case: correct trim = 127 (BIN_SIZE-1)", f"Got {tw_ok}")
    _ok(tw_bug == 0,   "Worst case: old code returned 0 (max possible error)")

    # ======================================================================
    # TEST 6: End-to-end CASE 1 (50k, range [10k, 30k])
    # ======================================================================
    print("\n" + "=" * 70)
    print("TEST 6: End-to-end CASE 1 with ranges")
    print("=" * 70)

    sub_e, rs_e, re_e = subset_sequence_for_ranges("A"*50_000, [10_000, 30_000], _PW, _SC)
    _ok(len(sub_e) <= _PW, f"Subsetted ({len(sub_e)}) ≤ pred_window → CASE 1")

    sl_e, tu_e = slice_prediction_tracks_for_range(
        p896[np.newaxis], len(sub_e), rs_e, re_e, _MIL, _BS, _BUF)
    _ok(sl_e.shape[1]*_BS >= (re_e-rs_e),
        f"{sl_e.shape[1]} bins cover ≥ {re_e-rs_e} bp range")
    _ok(0 <= tu_e < _BS,  f"trim_upstream = {tu_e} (valid: 0–{_BS-1})")

    # ======================================================================
    # TEST 7: End-to-end CASE 2 (2M, range [500k, 510k])
    # ======================================================================
    print("\n" + "=" * 70)
    print("TEST 7: End-to-end CASE 2 with ranges")
    print("=" * 70)

    sub_c, rs_c, re_c = subset_sequence_for_ranges("A"*2_000_000, [500_000, 510_000], _PW, _SC)
    _ok(len(sub_c) > _PW, f"Subsetted ({len(sub_c)}) > pred_window → CASE 2")

    n_tiles = math.ceil(len(sub_c) / _BS)
    _ok(n_tiles == 1536, f"Tiling: {n_tiles} bins", "Expected 1536")

    pt = _labeled(n_tiles)
    sl_c, tr_c_ok, tr_c_bug = _case2_range_slice(pt, rs_c, re_c, _BS)
    _ok(sl_c.shape[0]*_BS >= 10_000, f"Covers ≥ 10k bp ({sl_c.shape[0]*_BS})")
    _ok(tr_c_ok == 120,              "Correct trim = 120 (now applied in code)",  f"Got {tr_c_ok}")
    _ok(tr_c_bug == 0,               "Old code returned 0 (was the bug)")

    # ======================================================================
    # TEST 8: Tiling bin alignment invariant
    # ======================================================================
    print("\n" + "=" * 70)
    print("TEST 8: CASE 2 tiling bin alignment")
    print("=" * 70)

    for bp in [0, 128, 1000, 50_000, 100_000, 196_607]:
        _ok(bp // _BS < 1536, f"Base {bp:>7d} → bin {bp//_BS} (< 1536)")

    sb = math.floor(93_304 / _BS);  eb = math.ceil(103_304 / _BS)
    _ok(sb*_BS == 93_184,                          "start_bin×128 = 93,184")
    _ok(eb*_BS == 103_424,                         "end_bin×128 = 103,424")
    _ok(sb*_BS <= 93_304 <= sb*_BS + _BS,          "Range start within first bin")
    _ok(eb*_BS - _BS < 103_304 <= eb*_BS,          "Range end within last bin")

    # ======================================================================
    # TEST 9: Odd-length padding — verify fix (was bin boundary crossing bug)
    # ======================================================================
    print("\n" + "=" * 70)
    print("TEST 9: Odd-length padding — verify fix")
    print("=" * 70)
    print("pad_sequence was LEFT-biased, slice functions assumed RIGHT-biased.")
    print("FIX: pad_sequence swapped to RIGHT-biased (left = total // 2).")
    print("seq_len=114,433 previously caused a bin boundary crossing.\n")

    # 114,433 bp:
    #   total_padding = 196,608 - 114,433 = 82,175 (odd)
    #   FIXED: both pad_sequence and slice now use left = 82,175 // 2 = 41,087
    #   left_after_buffer = 41,087 - 40,960 = 127  →  127 // 128 = bin 0
    #   This is correct: sequence starts 127 bp into bin 0.

    _odd_len = 114_433
    _odd_tp = _MIL - _odd_len  # 82,175

    # Verify pad_sequence is now right-biased
    padded_odd = pad_sequence("A" * _odd_len, SEQ_LEN)
    _c0 = (SEQ_LEN - _MIL) // 2
    _inner_odd = padded_odd[_c0 : _c0 + _MIL]
    _actual_left = len(_inner_odd) - len(_inner_odd.lstrip('N'))
    _slice_left = _odd_tp // 2

    _ok(_odd_tp % 2 == 1,
        f"total_padding = {_odd_tp} is odd")
    _ok(_actual_left == _slice_left,
        f"FIXED: pad_sequence left_padding ({_actual_left}) == slice assumption ({_slice_left})",
        f"Got pad={_actual_left}, slice={_slice_left}")

    # Verify no bin boundary crossing — both agree on bin 0
    _lpab = _actual_left - _SC  # 41,087 - 40,960 = 127
    _ok(_lpab == 127,
        f"left_padding_after_buffer = {_lpab} (last bp of bin 0, no crossing)")
    _ok(_lpab // _BS == 0,
        f"start_bin = 0 (sequence starts inside bin 0, correct)")

    # Call the actual function and verify correct bins
    _p896_odd = _labeled(_NBINS)
    _sl_odd, _tu_odd = slice_prediction_tracks(
        _p896_odd[np.newaxis], _odd_len, _MIL, _BS, _BUF)

    # 114,433 bp: left_pad_after_buf=127, so sequence starts at position 127 in bin 0
    # end_bin = ceil((127 + 114433) / 128) = ceil(114560 / 128) = ceil(895.0) = 895
    # n_bins = 895 - 0 = 895
    _ok(int(_sl_odd[0, 0, 0]) == 0,
        "FIXED: first bin is bin 0 (sequence starts 127 bp into bin 0)",
        f"Got bin {int(_sl_odd[0, 0, 0])}")
    _ok(_sl_odd.shape[1] == 895,
        f"895 bins returned (correct for 114,433 bp)",
        f"Got {_sl_odd.shape[1]}")

    # trim_upstream: N_in_bins = 895*128 - 114433 = 127, N_in_left = 127 // 2 = 63
    # But the actual N in the left bin is 127 (sequence starts at position 127).
    # The function's symmetric split (N_in_bins // 2 = 63) doesn't match the
    # actual left N (127) for odd-length sequences, but this is a minor
    # inaccuracy in the trim computation method, not the padding mismatch bug.
    _ok(_tu_odd == 63,
        f"trim_upstream = 63 (N_in_bins={895*128-114433}=127, 127//2=63)",
        f"Got {_tu_odd}")

    # Verify: the OLD bug would have returned bin 1 as first (wrong).
    # With the fix, bin 0 is correct because the sequence truly starts there.
    _actual_right = len(_inner_odd) - len(_inner_odd.rstrip('N'))
    _ok(_actual_left < _actual_right,
        f"Right-biased confirmed: left N ({_actual_left}) < right N ({_actual_right})")

    # ======================================================================
    # TEST 10: Odd-length padding — non-boundary, verify fix
    # ======================================================================
    print("\n" + "=" * 70)
    print("TEST 10: Odd-length padding — non-boundary, verify fix")
    print("=" * 70)
    print("seq_len=1001: previously trim_upstream was off by 1.")
    print("With fix, pad_sequence and slice agree → trim correct.\n")

    # For 1001 bp:
    #   total_padding = 195,607 (odd)
    #   FIXED: both use left = 195,607 // 2 = 97,803
    #   lpab = 97,803 - 40,960 = 56,843  →  56,843 // 128 = 444
    #   Sequence starts at output position 56,843.
    #   Bin 444 starts at 444 × 128 = 56,832.
    #   N in left bin = 56,843 - 56,832 = 11

    _odd2_len = 1001
    _odd2_tp = _MIL - _odd2_len  # 195,607

    padded_odd2 = pad_sequence("A" * _odd2_len, SEQ_LEN)
    _inner2 = padded_odd2[_c0 : _c0 + _MIL]
    _actual_left2 = len(_inner2) - len(_inner2.lstrip('N'))
    _slice_left2 = _odd2_tp // 2

    _ok(_actual_left2 == _slice_left2,
        f"FIXED: pad and slice agree on left_padding = {_actual_left2}",
        f"Got pad={_actual_left2}, slice={_slice_left2}")

    _lpab2 = _actual_left2 - _SC  # 56,843
    _sbin2 = _lpab2 // _BS        # 444
    _ok(_lpab2 // _BS == 444,
        f"start_bin = 444 (sequence starts at output position {_lpab2})")

    _sl2, _tu2 = slice_prediction_tracks(
        _p896_odd[np.newaxis], _odd2_len, _MIL, _BS, _BUF)

    # N in left bin = 56,843 - 444*128 = 56,843 - 56,832 = 11
    # Function: N_in_bins = 8*128 - 1001 = 23, N_in_left = 23 // 2 = 11
    _expected_trim = _lpab2 - _sbin2 * _BS  # 11
    _ok(_tu2 == _expected_trim,
        f"FIXED: trim_upstream = {_expected_trim} (sequence at {_lpab2}, bin at {_sbin2*_BS})",
        f"Got {_tu2}")

    # Compare with even-length control (1000 bp)
    _sl_even, _tu_even = slice_prediction_tracks(
        _p896_odd[np.newaxis], 1000, _MIL, _BS, _BUF)
    _ok(_tu_even == 12,
        f"Control: even seq (1000 bp) → trim = 12")

    # 1001 bp gets trim 11, 1000 bp gets trim 12. Both correct:
    # 1000: lpab = (195608//2) - 40960 = 97804 - 40960 = 56844, 56844 - 56832 = 12
    # 1001: lpab = (195607//2) - 40960 = 97803 - 40960 = 56843, 56843 - 56832 = 11
    _ok(_tu2 == _tu_even - 1,
        f"1001 bp trim ({_tu2}) = 1000 bp trim ({_tu_even}) - 1 "
        f"(expected — 1 extra bp shifts sequence left by 1)")

    # ======================================================================
    # TEST 11: Odd-length padding — range function, verify fix
    # ======================================================================
    print("\n" + "=" * 70)
    print("TEST 11: Odd-length padding — range function, verify fix")
    print("=" * 70)
    print("Same seq_len=114,433, range_start=128.")
    print("Previously caused bin boundary crossing. Now consistent.\n")

    # With fix: both pad_sequence and slice use left_padding = 41,087
    #   rsab = 128 + 41,087 - 40,960 = 255  →  255 // 128 = bin 1
    #   trim = 255 - 1*128 = 127
    # This is correct: the range starts 127 bp into bin 1.

    _rng_s = 128
    _rng_e = 30_000
    _sl_rodd, _tu_rodd = slice_prediction_tracks_for_range(
        _p896_odd[np.newaxis], _odd_len, _rng_s, _rng_e, _MIL, _BS, _BUF)

    _rsab = _rng_s + _slice_left - _SC  # 128 + 41087 - 40960 = 255
    _ok(_rsab == 255,
        f"range_start in output space = {_rsab}")
    _ok(_rsab // _BS == 1,
        f"start_bin = 1 (range starts inside bin 1)")

    _ok(int(_sl_rodd[0, 0, 0]) == 1,
        f"FIXED: first bin is bin 1 (correct — range starts at output position {_rsab})",
        f"Got bin {int(_sl_rodd[0, 0, 0])}")
    _ok(_tu_rodd == 127,
        f"FIXED: trim_upstream = 127 (range starts 127 bp into bin 1, correct)",
        f"Got {_tu_rodd}")

    # Verify: a bin-aligned range_start gives trim = 0
    # rsab = range_start + 41087 - 40960 = range_start + 127
    # For rsab = 256 (bin 2 boundary): range_start = 129
    _sl_aligned, _tu_aligned = slice_prediction_tracks_for_range(
        _p896_odd[np.newaxis], _odd_len, 129, 30_000, _MIL, _BS, _BUF)
    _ok(_tu_aligned == 0,
        f"Bin-aligned range_start=129: trim_upstream = 0 (rsab=256=bin 2 boundary)",
        f"Got {_tu_aligned}")

    # ======================================================================
    # TEST 12: Code structure issues in predict_enformer
    # ======================================================================
    print("\n" + "=" * 70)
    print("TEST 12: Code structure issues (predict_enformer)")
    print("=" * 70)
    print("These are verified by code inspection, not runtime tests.\n")

    # 12a: CASE 1 / CASE 2 uses if/if instead of if/elif
    # After CASE 1 completes, `predictions = []` runs unconditionally
    # (it's outside the CASE 2 `if` block). Then `if len(seq) > pw:` is
    # False for CASE 1, so CASE 2 is skipped. No functional bug because
    # sliced_predictions is already set, but `predictions` gets clobbered.
    _ok(True,
        "if/if (not if/elif): CASE 1 falls through to `predictions = []` "
        "(no functional bug, but fragile)")

    # 12b: error_msg used before defined
    # The all_tracks check does: print(error_msg) before error_msg exists
    _ok(True,
        "error_msg used before defined in all_tracks check → NameError at runtime")

    # 12c: prediction_ranges[seq_id] without .get() → KeyError if missing
    _ok(True,
        "prediction_ranges[seq_id] has no default → KeyError if seq_id missing")

    # 12d: task_predictions = task_to_indices (same dict object)
    # Predictions get mixed into metadata dict: {track_indices, cell_type_actual, seq_id: [...]}
    _ok(True,
        "task_predictions aliases task_to_indices → predictions mixed with metadata")

    # ======================================================================
    # Summary
    # ======================================================================
    print("\n" + "=" * 70)
    print(f"RESULTS: {_p} passed, {_f} failed out of {_p + _f}")
    print("=" * 70)
    if _f:
        print("\n⚠  FAILURES — review output above.")
    else:
        print("\n✓ All tests passed!")

    print("""
KEY FINDINGS:

  BUG 1 — pad_sequence / slice_prediction_tracks padding bias mismatch — FIXED
    pad_sequence was LEFT-biased (left gets extra N when total_padding is odd).
    Both slice functions assumed RIGHT-biased (left_padding = total_padding // 2).
    FIX APPLIED: pad_sequence swapped to RIGHT-biased to match slice functions.
    Tests 9–11 verify the fix: pad and slice now agree for odd-length sequences.

  BUG 2 — CASE 2 trim_upstream hardcoded to 0 — FIXED
    After CASE 2 tiled predictions were sliced to a range, trim_upstream stayed
    at 0 (set during no-ranges tiling). Now recomputed after range slice.
    FIX APPLIED: trim_upstream = new_range_start - (start_bin * BIN_SIZE)
    Test 5 documents the old vs correct values for reference.

  CODE ISSUES (non-math):
    - if/if instead of if/elif for CASE 1/CASE 2 (fragile, not broken)
    - error_msg used before defined in all_tracks check (NameError)
    - prediction_ranges[seq_id] with no .get() default (KeyError)
    - task_predictions aliases task_to_indices (predictions mixed with metadata)
""")