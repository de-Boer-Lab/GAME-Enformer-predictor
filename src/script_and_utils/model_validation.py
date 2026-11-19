import numpy as np
from error_checking_functions import *

# Define what this specific model supports globally
SUPPORTED_SCALES = ["linear", "log"]
DEFAULT_SCALE = "linear"

def unsupported_task_reason(task):
    """
    Checks whether Enformer can perform a single prediction task, based on its species and type.

    A task Enformer cannot perform does NOT fail the whole request: it is returned with
    null `*_actual` keys and an error object in `predictions` (see Predictor API Schema),
    while the other tasks are still predicted.

    Args:
        task (dict): One object of `prediction_tasks` from the Evaluator request.

    Returns:
        str or None: A message describing why the task cannot be performed, or None if it can.
    """
    reasons = []

    # --- MODEL SPECIFIC: Ensure this Enformer Predictor only supports homo_sapiens and mus_musculus ---
    if task.get('species', '').lower() not in ["homo_sapiens", "mus_musculus"]:
        reasons.append(
            f"This predictor only supports species: ['homo_sapiens', 'mus_musculus']. Received '{task.get('species')}'."
        )
    task_type = task.get('type', '').lower()
    if task_type.startswith("conformation_") or task_type.startswith("expression_splicing"):
        reasons.append(
            f"This predictor only supports type: ['expression', 'expression_pol1', 'expression_pol2', 'expression_pol3', 'expression_mrna', 'accessibility', 'binding_[molecule]']. Received '{task.get('type')}'."
        )

    return " ".join(reasons) if reasons else None

def model_specific_payload_validation(payload):
    """
    Request-level Enformer checks. These fail the whole request (HTTP error).
    Task-level limitations (species, type) are handled per task by `unsupported_task_reason`.
    """

    errors = {'prediction_request_failed': []}

    readout_type = payload['readout']

    # Handle unsupported `interaction_matrix` readout
    if readout_type == "interaction_matrix":
        print("Enformer cannot handle 'interaction_matrix' readout type. Exiting gracefully!")
        errors['prediction_request_failed'].append("Enformer cannot process 'interaction_matrix' readout type.")

    for task in payload['prediction_tasks']:
        # --- MODEL-SPECIFIC: Determine the scale of the prediction requested ---
        # --- NOTE: Commented out old code. Enformer supports linear or log ---
        # For now, Enformer only supports scale_prediction_requested: "linear"
        # This can change for other Predictors, in which case they should remap scale_prediction_actual
        # and specify explicitly what base they are using, if scale is logarithmic.
        # for task in payload['prediction_tasks']:
            # if 'scale' not in task:
            #     print(f"No scale value provided for task '{task.get('name')}'. Defaulting to 'linear'.")
            # elif task.get('scale', '').lower() not in ["linear"]:
            #     print(f"This predictor only supports scale_prediction: ['linear']. Received '{task.get('scale')}' for task '{task.get('name')}'.")
            #     errors['prediction_request_failed'].append(
            #         f"This predictor only supports scale_prediction: ['linear']. Received '{task.get('scale')}' for task '{task.get('name')}'."
            #     )
        # --- OLD CODE ENDS ---
        
        # This can change for other Predictors, in which case they should remap scale_prediction_actual
        # and specify explicitly what base they are using, if scale is logarithmic.
        # Must be 'linear' or 'log'.
        req_scale = task.get('scale')
        if req_scale and req_scale.lower() not in SUPPORTED_SCALES:
            errors['prediction_request_failed'].append(
                f"Unsupported scale: '{req_scale}'. Supported scales are: {SUPPORTED_SCALES}."
            )
    
    #If you want to add error checking that restricts sequences with N bases, add that here
    if any(errors.values()):
        flagged_errors = [msg for sublist in errors.values() for msg in sublist]
        raise PredictionFailedError(flagged_errors)
    
def apply_scaling(predictions_dict, requested_scale):
    """
    Applies scaling transformation specific to THIS model's output and returns the applied scale name.
    
    Enformer Default Output: Linear
    Logic: 
      - If 'linear' requested: Do nothing.
      - If 'log' requested: log2(x + 1)
      
    Args:
        predictions_dict (dict): The raw linear predictions
        requested_scale (str or None): The scale requested by the user
        
    Returns:
        tuple: (transformed_dict, actual_scale_str)
    """
    
    # Determine Effective Scale
    if not requested_scale:
        # Default if None provided
        effective_scale = DEFAULT_SCALE
    else:
        effective_scale = requested_scale.lower()
    
    if effective_scale == "linear":
        return predictions_dict, "linear"
    
    transformed_preds = {}
    for seq_id, values in predictions_dict.items():
        # Convert to numpy for fast vectorized math
        arr = np.array(values)
        
        if effective_scale == "log":
            arr = np.log2(arr + 1)
        
        # Convert back to list for JSON serialization
        transformed_preds[seq_id] = arr.tolist()
        
    return transformed_preds, effective_scale
    
