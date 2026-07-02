# api_preprocessing_utils.py
import math

def pad_sequence(seq, target_length):
    """
    Pad a sequence, without adapters, with 'N' until it reached the target length.

    Args:
        seq (str): The input sequence.
        target_length (int): Model-dependent desired length of the sequence before
                             adding adapters.

    Returns:
        padded_seq: The padded sequence.
    """
    seq_len = len(seq)

    # If sequence length is less than target_length, excluding adapters, simply
    # pad with Ns until it is target_length and then add the adapters.
    # NOTE: Swapped this to be right biased padding to match slicing logic.
    if seq_len < target_length:
        total_padding = target_length - seq_len
        left_padding = 'N' * (total_padding // 2)
        right_padding = 'N' * (total_padding - len(left_padding))
        padded_seq = left_padding + seq + right_padding
        return padded_seq

    # When the sequence length, excluding adapters, is the target_length,
    # simply add adapters to each side.
    elif seq_len == target_length:
        return seq


def add_adapters(seq, upstream_adapter, downstream_adapter):
    """
    Add upstream and downstream adapter sequences to a given sequence.

    Args:
        seq (str): The input sequence after padding, if needed.
        upstream_adapter (str): Upstream adapter sequence to prepend.
        downstream_adapter (str): Downstream adapter sequence to append.

    Returns:
        seq_with_adapters (str): The sequence with adapters added.
    """
    seq_with_adapters = upstream_adapter + seq + downstream_adapter
    return seq_with_adapters


# Full preprocessing pipeline for a sequence
def process_sequence(seq, target_length):
    """
    Process a sequence by padding, adding adapters, and encoding.

    Args:
        seq (str): The input sequence.
        target_length (int): Length of the sequence before adding adapters (200 for Dream-RNN).
        seq_size (int): Model-specific final sequence size (230 for Dream-RNN)
        upstream_adapter (str): Upstream adapter sequence to prepend.
        downstream_adapter (str): Downstream adapter sequence to append.

    Returns:
        list: A one-hot encoded list with padding and adapters applied.
    """
    # Step 1: Pad the sequence
    padded_seq = pad_sequence(seq, target_length)

    return padded_seq

def subset_sequence_for_ranges(sequence, pred_range, prediction_window, context_flank):
    """
    Subset the input sequence based on the prediction range, prediction window, and context flank.
    
    It means we will take a portion of the input sequence that includes the prediction range 
    and additional context on either side, based on the specified prediction window and context flank.
    This is done to ensure that the model has enough surrounding sequence information to make accurate 
    predictions for the specified range.
    
    Logic:
    1. If prediction range size < prediction window:
        center the prediction range within the prediction window 
        and add context flank on either side.
    2. If prediction range size >= prediction window:
        simply add context flank on either side of the prediction range.

    Args:
        sequence (str): The input sequence to be subsetted.
        pred_range (list): The start and end indices of the prediction range within the sequence.
        prediction_window (int): The desired size of the prediction window around the prediction range.
        context_flank (int): The number of additional bases to include on either side of the prediction window for context.

    Returns:
        tuple: A tuple containing the subsetted sequence, 
               the new start index of the prediction range within the subsetted sequence,
               and the new end index of the prediction range within the subsetted sequence.
    """
    start, end = pred_range # Unpack the prediction range into start and end indices
    pred_range_size = end - start # Calculate the size of the prediction range

    if pred_range_size < prediction_window:
        print("Subsetting for range size < 114kb")
        pred_range_mid = (end + start)/2
        #Use floor to left side to not loose bases
        new_start = max(math.floor(pred_range_mid - prediction_window/2 - context_flank), 0)
        #use ceil on right side to not loose bases
        new_end = min(math.ceil(pred_range_mid + prediction_window/2 + context_flank), len(sequence)) 
        
    else:
        print("Subsetting for range size >= 114kb")
        new_start = max(math.floor(start - context_flank), 0)
        new_end = min(math.ceil(end + context_flank), len(sequence))
    
    sequence_subsetted = sequence[new_start:new_end]

    #The prediction range start and end is now shifted with respect to the subsetting
    new_range_start = start - new_start
    new_range_end = end - new_start
    print("New ranges are")
    print(new_range_start)
    print(new_range_end)
    return sequence_subsetted, new_range_start, new_range_end

