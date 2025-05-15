# config.py
import os
import json
import sys

class Config:
    def __init__(self, filepath="settings.json"):
        self.settings = {}
        self.filepath_for_error_reporting = filepath
        self._load_config(filepath)
        self._validate_and_initialize()

    def _load_config(self, filepath):
        """Loads configuration from a JSON file."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                self.settings = json.load(f)
        except FileNotFoundError:
            print(f"FATAL ERROR: Configuration file '{filepath}' not found. Please create it.")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"FATAL ERROR: Could not parse JSON file '{filepath}': {e}")
            sys.exit(1)
        except Exception as e:
            print(f"FATAL ERROR: Reading config file '{filepath}': {e}")
            sys.exit(1)

    def _validate_and_initialize(self):
        """Validates required keys and sets them as attributes with type checking."""
        expected_keys_and_types = {
            "WORKER_API_URL": str, "WORKER_API_TOKEN": str, "WORKER_MODEL_IDS": "list_str",
            "ACCURACY_JUDGE_API_URL": str, "ACCURACY_JUDGE_API_TOKEN": str, "ACCURACY_JUDGE_MODEL_ID": str,
            "INTEGRITY_JUDGE_API_URL": str, "INTEGRITY_JUDGE_API_TOKEN": str, "INTEGRITY_JUDGE_MODEL_ID": str,
            
            "DATASET_CONFIGS": dict, 
            "DATASETS_TO_RUN": "list_str", 

            "WORKER_OUTPUT_FILE_TEMPLATE": str, "FINAL_OUTPUT_FILE_TEMPLATE": str,
            "SKIPPED_FILE_LOG_TEMPLATE": str, "SUMMARY_FILE_TEMPLATE": str,
            "PROMPT_VERSIONS_TO_TEST": "list_str",
            "TOKEN_BUDGET_EFFICIENCY": int, "P_IRRELEVANT_EFFICIENCY": float,
            "SAFETY_SEVERE_KEYWORDS": str, 
            "ALIGNMENT_LENGTH_MISMATCH_PENALTY": int, "ALIGNMENT_MAX_LENGTH_RATIO_VS_REF": float,
            "WEIGHT_ACCURACY": float, "WEIGHT_TRUE_INTEGRITY": float, "WEIGHT_EFFICIENCY": float,
            "WEIGHT_SAFETY": float, "WEIGHT_ALIGNMENT_SIMPLE": float,
            "MAX_RETRIES": int, "RETRY_DELAY_SECONDS": int, "REQUEST_TIMEOUT_SECONDS": int,
            "MAX_CONCURRENT_ITEMS_PER_COMBO": int,
            "OPENROUTER_API_BASE_URL": str, "OPENROUTER_API_KEY": str,
            "OPENROUTER_HTTP_REFERER": str, "OPENROUTER_X_TITLE": str
        }
        
        all_required_keys = list(expected_keys_and_types.keys())
        actual_settings_keys = {k for k in self.settings if not k.startswith("_comment_")}
        missing_keys = [key for key in all_required_keys if key not in actual_settings_keys]

        if missing_keys:
            print(f"FATAL ERROR: Missing required keys in '{self.filepath_for_error_reporting}': {', '.join(missing_keys)}")
            sys.exit(1)

        for key, expected_type_or_str in expected_keys_and_types.items():
            value = self.settings[key]
            valid_type = False
            if expected_type_or_str == str:
                if isinstance(value, str): valid_type = True
                elif key == "SAFETY_SEVERE_KEYWORDS" and value is None: value = ""; valid_type = True # Allow null for SAFETY_KEYWORDS, default to empty
            elif expected_type_or_str == int:
                if isinstance(value, int): valid_type = True
            elif expected_type_or_str == float:
                if isinstance(value, (int, float)): value = float(value); valid_type = True
            elif expected_type_or_str == "list_str":
                # WORKER_MODEL_IDS and PROMPT_VERSIONS_TO_TEST must be non-empty
                if key in ["WORKER_MODEL_IDS", "PROMPT_VERSIONS_TO_TEST"]:
                    if isinstance(value, list) and all(isinstance(item, str) for item in value) and value: valid_type = True
                elif key == "DATASETS_TO_RUN": # Can be empty list
                     if isinstance(value, list) and all(isinstance(item, str) for item in value): valid_type = True
            elif expected_type_or_str == dict: 
                 if isinstance(value, dict): valid_type = True
            
            if not valid_type:
                expected_type_name = expected_type_or_str if isinstance(expected_type_or_str, str) else expected_type_or_str.__name__
                print(f"FATAL ERROR: For key '{key}', expected type '{expected_type_name}', got {type(value)} (value: '{value}'). Check '{self.filepath_for_error_reporting}'.")
                sys.exit(1)
            setattr(self, key, value)

            if (key.endswith("_API_TOKEN") or key == "OPENROUTER_API_KEY") and isinstance(value, str) and \
               any(placeholder in value.lower() for placeholder in ["your_", "_here"]):
                print(f"WARNING: API Token/Key for '{key}' in '{self.filepath_for_error_reporting}' appears to be a placeholder: '{value}'. Please update.")
        
        if not isinstance(self.DATASET_CONFIGS, dict): # Should be caught by type check above, but as safeguard
            print(f"FATAL ERROR: DATASET_CONFIGS must be a dictionary in '{self.filepath_for_error_reporting}'.")
            sys.exit(1)
        for ds_short_name_to_run in self.DATASETS_TO_RUN: # Validate only datasets selected to run
            if ds_short_name_to_run not in self.DATASET_CONFIGS:
                print(f"FATAL ERROR: Dataset short name '{ds_short_name_to_run}' in DATASETS_TO_RUN is not defined in DATASET_CONFIGS. Check '{self.filepath_for_error_reporting}'.")
                sys.exit(1)
            ds_config_value = self.DATASET_CONFIGS[ds_short_name_to_run]
            if not isinstance(ds_config_value, dict) or \
               "path" not in ds_config_value or \
               not isinstance(ds_config_value["path"], str):
                print(f"FATAL ERROR: Dataset configuration for '{ds_short_name_to_run}' in DATASET_CONFIGS is invalid. Must be a dict with a 'path' (string). Check '{self.filepath_for_error_reporting}'.")
                sys.exit(1)

        self.SAFETY_SEVERE_KEYWORDS_LIST = [kw.strip().lower() for kw in self.SAFETY_SEVERE_KEYWORDS.split(',') if kw.strip()] if self.SAFETY_SEVERE_KEYWORDS else []
        self.esi_weights = {
            "accuracy": self.WEIGHT_ACCURACY, "true_integrity": self.WEIGHT_TRUE_INTEGRITY,
            "efficiency": self.WEIGHT_EFFICIENCY, "safety": self.WEIGHT_SAFETY,
            "alignment_simple": self.WEIGHT_ALIGNMENT_SIMPLE
        }
        total_weight = sum(self.esi_weights.values())
        if not (abs(total_weight - 1.0) < 1e-9) and total_weight > 0: 
            print(f"INFO: ESI weights sum to {total_weight:.4f}. Normalizing to 1.0.")
            for k_weight in self.esi_weights: self.esi_weights[k_weight] /= total_weight
        elif total_weight <= 0: 
            print(f"FATAL ERROR: ESI weights must sum to a positive value. Sum: {total_weight:.4f}. Check '{self.filepath_for_error_reporting}'.")
            sys.exit(1)

APP_CONFIG = Config()

def _ensure_base_dir_from_template(template_str_attr_name_on_config: str):
    if hasattr(APP_CONFIG, template_str_attr_name_on_config):
        template_str = getattr(APP_CONFIG, template_str_attr_name_on_config)
        if isinstance(template_str, str):
            try:
                sample_path = template_str.format(dataset_short_name="testds", model_id="testmodel", prompt_version="testprompt")
                base_dir = os.path.dirname(sample_path)
            except KeyError: 
                base_dir = os.path.dirname(template_str)
            
            if base_dir and not os.path.exists(base_dir): # Ensure base_dir is not empty string
                try:
                    os.makedirs(base_dir, exist_ok=True)
                except OSError as e: 
                    print(f"WARNING: Could not create base directory '{base_dir}' from template key '{template_str_attr_name_on_config}': {e}")

_ensure_base_dir_from_template('WORKER_OUTPUT_FILE_TEMPLATE')
_ensure_base_dir_from_template('FINAL_OUTPUT_FILE_TEMPLATE')
_ensure_base_dir_from_template('SKIPPED_FILE_LOG_TEMPLATE')
_ensure_base_dir_from_template('SUMMARY_FILE_TEMPLATE')

if hasattr(APP_CONFIG, 'DATASET_CONFIGS') and isinstance(APP_CONFIG.DATASET_CONFIGS, dict):
    for ds_config_val in APP_CONFIG.DATASET_CONFIGS.values(): # Iterate through values of the dict
        if isinstance(ds_config_val, dict) and "path" in ds_config_val and isinstance(ds_config_val["path"], str):
            input_file_path = ds_config_val["path"]
            base_input_dir = os.path.dirname(input_file_path)
            if base_input_dir and not os.path.exists(base_input_dir) : 
                 try:
                    os.makedirs(base_input_dir, exist_ok=True)
                 except OSError as e:
                    print(f"WARNING: Could not create base input directory '{base_input_dir}' for path '{input_file_path}': {e}")