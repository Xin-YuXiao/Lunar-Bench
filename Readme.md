# Comprehensive LLM Evaluation Framework (ESI & Contextual ACC)

## 1. Overview

This framework provides a robust system for evaluating Large Language Models (LLMs) by generating responses to a given dataset and then assessing these responses across multiple dimensions. It uniquely features:

* **Multi-Model & Multi-Prompt Testing**: Simultaneously evaluate various worker LLMs using different prompting strategies (e.g., Direct, Chain-of-Thought).
* **Concurrent Pipelined Processing**: For each model-prompt combination, data items are processed through a concurrent pipeline (Worker LLM call -> Answer Cleaning -> Accuracy Judge call -> Integrity Judge call -> ESI Score calculation), maximizing efficiency for I/O-bound API calls.
* **Context-Dependent Accuracy (ACC)**: Employs distinct judging criteria for accuracy based on the input dataset type (L1-1K, L2-1K, L3-1K), allowing for nuanced evaluation from very lenient to strict.
* **True Integrity Score**: Assesses the completeness of the worker LLM's output and its thought process, specifically whether all conditions and aspects of the input were considered.
* **ESI Score Calculation**: Computes an overall ESI (Efficiency, Safety, Integrity (True), Alignment (Simplified)) score, providing a multi-faceted view of model performance.
* **Detailed Reporting**: Generates detailed JSONL output files for each test run and a JSON summary file with aggregated metrics.
* **High Configurability**: All critical parameters, including API endpoints, tokens, model IDs, file paths, ESI weights, and evaluation criteria, are managed through a central `settings.json` file.

This framework is designed to move beyond simple accuracy metrics and provide a more holistic understanding of LLM capabilities.

## 2. Core Concepts & Definitions

### 2.1. LLM Roles

* **Worker LLM**: The primary LLM being evaluated. It generates answers based on the provided `instruction`, `question`, and a specific `prompt_version` (e.g., DIRECT, COT). Multiple Worker LLMs can be specified for comparative evaluation.
* **Accuracy Judge LLM**: A separate LLM (e.g., `deepseek-chat`) tasked with evaluating the correctness of the Worker LLM's final cleaned answer. Its strictness and criteria are determined by the type of input file being processed (L1, L2, or L3).
* **Integrity Judge LLM**: Another LLM (can be the same model as the Accuracy Judge but with a different prompt) that evaluates the *process completeness* and *condition coverage* of the Worker LLM's output (including raw output which might contain reasoning steps). It assigns a score from 0-100 for "True Integrity."

### 2.2. Evaluation Metrics

The framework calculates the following key metrics for each answer:

1.  **ACC (Accuracy Score - $S_{accuracy}$)**:
    * **Definition**: Measures whether the Worker LLM's final cleaned answer is acceptably correct in addressing the core of the `Question`, considering the `Instruction` and `Reference Answer`.
    * **Judgment**: Performed by the Accuracy Judge LLM. The judgment criteria vary based on the input file:
        * **L1-1K (Very Lenient)**: The answer is judged correct if it's "approximately correct or on the right track." Focus is on capturing the gist, even with imperfections. (Uses `PROMPT_FOR_JUDGE_L1_ACCURACY_TEMPLATE`).
        * **L2-1K (Balanced/Reasonable)**: The answer is judged correct if it's "acceptably correct and appropriate," addressing the main intent and containing key factual information without major errors. (Uses `PROMPT_FOR_JUDGE_L2_ACCURACY_TEMPLATE`).
        * **L3-1K (Stricter, Logical)**: The answer is judged correct if it's "factually correct, logically sound, and precisely addresses the Question." (Uses `PROMPT_FOR_JUDGE_L3_ACCURACY_TEMPLATE`).
    * **Output**: The Judge LLM provides a boolean (`is_judged_correct`). The $S_{accuracy}$ is 100.0 if true, 0.0 if false.

2.  **True Integrity Score ($S_{true\_integrity}$)**:
    * **Definition**: Measures the completeness and logical integrity of the Worker LLM's output and (if visible) its thought process. It assesses whether all relevant conditions from the `Instruction` and `Question` were considered. This is distinct from the final answer's factual accuracy.
    * **Judgment**: Performed by the Integrity Judge LLM using `PROMPT_FOR_JUDGE_LLM_TRUE_INTEGRITY_TEMPLATE`.
    * **Output**: The Judge LLM provides a numerical `integrity_score` (0-100). This score directly becomes $S_{true\_integrity}$.

3.  **Efficiency Score ($S_{eff}$)**:
    * **Definition**: Measures the token efficiency of the Worker LLM's generated answer.
    * **Formula**: $S_{eff} = S_{budget} \times (1 - P_{irr})$
        * $S_{budget} = \max(0, 1 - T_{used} / \text{TOKEN\_BUDGET\_EFFICIENCY}) \times 100$
        * $T_{used}$ is the number of completion tokens used by the Worker LLM.
        * `TOKEN_BUDGET_EFFICIENCY` (e.g., 8000) is a configurable budget from `settings.json`.
        * $P_{irr}$ is the proportion of irrelevant tokens, configurable via `P_IRRELEVANT_EFFICIENCY` in `settings.json` (currently defaults to 0, meaning all worker completion tokens are considered relevant for the $S_{budget}$ calculation if not otherwise specified).

4.  **Safety Score ($S_{safety}$)**:
    * **Definition**: A basic measure to detect potentially harmful or problematic content based on keywords. It acts as a critical gate.
    * **Formula (Conceptual)**: Aligns with $S_{safety} = 100 \times \Pi(\neg \text{DetectSevereRisk}(\text{Output, Protocol}))$ where `DetectSevereRisk` is implemented by checking for keywords.
    * **Judgment**: If any keyword from `SAFETY_SEVERE_KEYWORDS` (defined in `settings.json`) is found in the `worker_answer_cleaned`, $S_{safety}$ is 0. Otherwise, it is 100.
    * **Impact on ESI**: If $S_{safety}$ is 0, the final ESI score for that item is also forced to 0.

5.  **Simplified Alignment Score ($S_{align\_simple}$)**:
    * **Definition**: A simplified proxy for alignment in a single-turn Q&A context. It considers if the answer was accurate, followed expected formatting (for CoT prompts), and was not excessively verbose. This is an adaptation and does not cover deeper ethical or value alignment from the image shown initially.
    * **Calculation**: Starts at 100 and deducts points if:
        * The answer is not accurate (based on ACC judgment).
        * For `COT` prompts, if the "Final Answer:" marker is missing.
        * The cleaned answer's length rapporto to the reference answer exceeds `ALIGNMENT_MAX_LENGTH_RATIO_VS_REF` (penalty: `ALIGNMENT_LENGTH_MISMATCH_PENALTY`).

6.  **ESI Score (Overall Evaluation Score)**:
    * **Definition**: A weighted sum of the above individual metrics.
    * **Formula**: $ESI = (w_{ACC} \cdot S_{accuracy}) + (w_{TI} \cdot S_{true\_integrity}) + (w_{Eff} \cdot S_{eff}) + (w_{Safe} \cdot S_{safety}) + (w_{AlignS} \cdot S_{align\_simple})$
    * Weights ($w_{ACC}$, etc.) are configurable in `settings.json` via `WEIGHT_...` keys and are normalized to sum to 1.0 by the script.

## 3. Workflow

The evaluation process for each specified Worker Model and Prompt Version combination proceeds as follows:

1.  **Initialization**:
    * The `main.py` script is executed.
    * Global configurations are loaded from `settings.json` via `config.py`.
    * The input data file (e.g., `L1-1K.jsonl`) is read into memory.
    * The appropriate ACCURACY Judge prompt is selected based on the `INPUT_FILE` name.

2.  **Per Combination Processing**:
    The script iterates through each `worker_model_id` in `WORKER_MODEL_IDS` and then through each `prompt_version` in `PROMPT_VERSIONS_TO_TEST`. For each combination:
    * A dedicated output file, skipped log, and summary file are prepared.
    * A `ThreadPoolExecutor` is created to manage concurrent processing of individual data items from the input file. The number of concurrent items is set by `MAX_CONCURRENT_ITEMS_PER_COMBO`.
    * A `tqdm` progress bar is displayed for this specific combination.

3.  **Per Data Item Processing (Concurrent Pipeline in Threads)**:
    Each item from the input file is processed by a separate thread executing the `process_single_item_full_pipeline` function:
    * **a. Load Item Data**: Instruction, question, reference answer, etc., are parsed.
    * **b. Worker LLM Call**:
        * The appropriate Worker LLM prompt (Direct, CoT, Expert) is formatted.
        * An API call is made to the configured `WORKER_API_URL` using the `WORKER_API_TOKEN` and current `worker_model_id`.
        * The raw answer (`worker_answer_raw`), token usage, and response time are recorded.
    * **c. Answer Cleaning**: The `worker_answer_raw` is cleaned using `utils.clean_worker_model_answer` to produce `worker_answer_cleaned`. CoT format adherence is also checked.
    * **d. Accuracy (ACC) Judge Call**:
        * The selected L1/L2/L3-specific ACCURACY Judge prompt is formatted.
        * An API call is made to `ACCURACY_JUDGE_API_URL` using `ACCURACY_JUDGE_API_TOKEN` and `ACCURACY_JUDGE_MODEL_ID`.
        * The judge returns a boolean (`is_judged_correct`) and reasoning.
    * **e. True Integrity Judge Call**:
        * The True Integrity Judge prompt is formatted using `worker_answer_raw` (to see potential reasoning) and other context.
        * An API call is made to `INTEGRITY_JUDGE_API_URL` using `INTEGRITY_JUDGE_API_TOKEN` and `INTEGRITY_JUDGE_MODEL_ID`.
        * The judge returns a numerical `integrity_score` (0-100) and reasoning.
    * **f. ESI Sub-Score Calculation**:
        * $S_{accuracy}$ is calculated based on `is_judged_correct`.
        * $S_{true\_integrity}$ is derived from the `integrity_score`.
        * $S_{efficiency}$ is calculated from worker completion tokens.
        * $S_{safety}$ is calculated based on keyword detection.
        * $S_{align\_simple}$ is calculated.
    * **g. Final ESI Score Calculation**: The weighted ESI score is computed. If $S_{safety}$ is 0, ESI is forced to 0.
    * **h. Result Aggregation**: All data, LLM outputs, judge verdicts, and scores for this item are compiled into a dictionary.

4.  **Result Collection & Output**:
    * The main thread collects results from all completed futures (threads).
    * Collected results are written to the combination-specific detailed output JSONL file.
    * The `tqdm` progress bar is updated with average ACC, ESI, and error counts for the current combination.

5.  **Combination Summary**:
    * After all items for a combination are processed, a summary report is printed to the console.
    * A JSON summary file for the combination is saved, containing average metrics and processing statistics.

6.  **Loop**: The process repeats for the next model/prompt combination until all are done.

## 4. Project Structure
Your_Project_Root_Directory/
├── settings.json                 # Configuration file for all settings
├── config.py                     # Loads and validates settings.json
├── prompts.py                    # Contains all LLM prompt templates
├── llm_calls.py                  # Handles all API interactions with LLMs
├── utils.py                      # Utility functions (e.g., text cleaning)
├── evaluation_metrics.py         # Logic for calculating ACC, True Integrity, and ESI sub-scores
├── main.py                       # Main execution script (orchestrates the pipeline)
├── Data/                           # Directory for input .jsonl data files
│   └── demo.jsonl                # Example input file (or L1-1K.jsonl, etc.)
├── Intermediate/                 # (Optional) If two-stage execution is used for worker outputs
│   └── WorkerOutput_...jsonl
└── Result/                       # Directory for all output files
├── ESI_Result_...jsonl       # Detailed results for each combination
├── Summary_...json           # Summary statistics for each combination
└── Skipped_Log_...txt        # Log of skipped/errored items for each combination

## 5. Setup Instructions

1.  **Prerequisites**:
    * Python 3.7+ (recommended).
2.  **Download/Place Files**:
    * Ensure all Python files (`config.py`, `prompts.py`, `llm_calls.py`, `utils.py`, `evaluation_metrics.py`, `main.py`) are in your main project directory.
    * Create `settings.json` in the same directory.
3.  **Create Directories**:
    * In your project directory, create a `Data/` subdirectory. Place your input `.jsonl` files (e.g., `demo.jsonl`, `L1-1K.jsonl`) here.
    * The `Result/` and `Intermediate/` directories will be created automatically by `config.py` if they don't exist, based on the paths in `settings.json`.
4.  **Install Dependencies**:
    Open your terminal or command prompt and run:
    ```bash
    pip install requests tqdm
    ```
5.  **Configure `settings.json`**:
    This is the most crucial step. Open `settings.json` and carefully update the following:
    * **API Tokens**:
        * `WORKER_API_TOKEN`: Your API key for the Worker LLM service (e.g., SiliconFlow).
        * `ACCURACY_JUDGE_API_TOKEN`: Your API key for the Accuracy Judge LLM service (e.g., DeepSeek).
        * `INTEGRITY_JUDGE_API_TOKEN`: Your API key for the Integrity Judge LLM service (e.g., DeepSeek).
    * **API URLs**:
        * `WORKER_API_URL`: Endpoint for the Worker LLM.
        * `ACCURACY_JUDGE_API_URL`: Endpoint for the Accuracy Judge.
        * `INTEGRITY_JUDGE_API_URL`: Endpoint for the Integrity Judge.
    * **Model IDs**:
        * `WORKER_MODEL_IDS`: A list of strings, e.g., `["internlm/internlm2_5-20b-chat", "Qwen/Qwen2.5-72B-Instruct"]`.
        * `ACCURACY_JUDGE_MODEL_ID`: e.g., `"deepseek-chat"`.
        * `INTEGRITY_JUDGE_MODEL_ID`: e.g., `"deepseek-chat"`.
    * **File Paths**:
        * `INPUT_FILE`: Path to the specific input dataset you want to process (e.g., `"./Data/L1-1K.jsonl"`). The filename (L1, L2, or L3) determines the ACCURACY judging strictness.
        * Verify `WORKER_OUTPUT_FILE_TEMPLATE`, `FINAL_OUTPUT_FILE_TEMPLATE`, `SKIPPED_FILE_LOG_TEMPLATE`, `SUMMARY_FILE_TEMPLATE` if you wish to change the default output locations/names.
    * **Prompt Versions**:
        * `PROMPT_VERSIONS_TO_TEST`: List of worker prompt strategies, e.g., `["DIRECT", "COT"]`.
    * **ESI & Metric Parameters**:
        * Review `TOKEN_BUDGET_EFFICIENCY`, `P_IRRELEVANT_EFFICIENCY`.
        * `SAFETY_SEVERE_KEYWORDS`: Comma-separated list of keywords; leave empty `""` if no keyword filtering is desired.
        * `ALIGNMENT_LENGTH_MISMATCH_PENALTY`, `ALIGNMENT_MAX_LENGTH_RATIO_VS_REF`.
        * `WEIGHT_ACCURACY`, `WEIGHT_TRUE_INTEGRITY`, `WEIGHT_EFFICIENCY`, `WEIGHT_SAFETY`, `WEIGHT_ALIGNMENT_SIMPLE`: Adjust these weights as per your evaluation priorities. The script normalizes them if their sum is not 1.0.
    * **API Call & Concurrency Settings**:
        * `MAX_RETRIES`, `RETRY_DELAY_SECONDS`, `REQUEST_TIMEOUT_SECONDS`.
        * `MAX_CONCURRENT_ITEMS_PER_COMBO`: Number of data items to process in parallel for each model/prompt combination. Adjust based on your API rate limits and system resources (e.g., 5-10 is a common starting point).

## 6. How to Run the Evaluation

1.  Navigate to your project's root directory in your terminal or command prompt.
2.  Execute the `main.py` script:
    ```bash
    python main.py
    ```
    The script will:
    * Load settings.
    * Identify the ACCURACY judge prompt based on `INPUT_FILE` in `settings.json`.
    * Iterate through each specified Worker Model and Prompt Version.
    * For each combination, it will concurrently process all items from the `INPUT_FILE` using the configured number of threads (`MAX_CONCURRENT_ITEMS_PER_COMBO`).
    * A progress bar will be displayed for each combination, showing progress, average ESI, average ACC, and error counts.
    * Detailed results and a summary will be saved for each combination.

    *(Note: The `--stage` argument for two-phase execution has been removed in favor of the default concurrent pipeline model where worker and judge calls for an item are pipelined within each thread.)*

## 7. Output Description

For each `worker_model_id` and `prompt_version` combination, the following files are generated (filenames include sanitized model ID and prompt version):

1.  **Detailed Results File (`FINAL_OUTPUT_FILE_TEMPLATE`)**:
    * e.g., `./Result/ESI_Result_internlm__internlm2_5-20b-chat_DIRECT.jsonl`
    * A JSONL file (each line is a JSON object).
    * Each line corresponds to one item from the input dataset and contains:
        * Original input data (`id`, `instruction`, `question`, `reference_answer`, `scenario_code`).
        * Worker LLM details (`worker_model_id`, `worker_prompt_version`, `worker_answer_raw`, `worker_answer_cleaned`, `worker_prompt_tokens`, `worker_completion_tokens`, `worker_response_time_seconds`, `worker_output_correctly_formatted`).
        * Accuracy Judge details (`accuracy_judge_model_id`, `judge_verdict_is_correct` (boolean from the L1/L2/L3 specific judge), `accuracy_judge_reasoning`, `accuracy_judge_response_time_seconds`).
        * Integrity Judge details (`integrity_judge_model_id`, `integrity_judge_score` (0-100), `integrity_judge_reasoning`, `integrity_judge_response_time_seconds`).
        * Calculated ESI sub-scores (`s_accuracy`, `s_true_integrity`, `s_efficiency`, `s_safety`, `s_alignment_simple`).
        * Final `esi_score`.
        * `status` field indicating processing outcome (e.g., "COMPLETED", "ERROR\_WORKER\_API").
        * `processing_error_details` if any error occurred for that item.

2.  **Summary File (`SUMMARY_FILE_TEMPLATE`)**:
    * e.g., `./Result/Summary_internlm__internlm2_5-20b-chat_DIRECT.json`
    * A JSON file containing:
        * `combination_details`: Worker model and prompt version.
        * `processing_summary`: Total items, items scored, error counts for worker and judges.
        * `metrics_summary`: Average scores for ACC, True Integrity, Efficiency, Safety, Simplified Alignment, and the overall ESI score for that combination. Also includes average API response times.
        * Paths to the `final_output_file` and `skipped_items_log`.

3.  **Skipped Log File (`SKIPPED_FILE_LOG_TEMPLATE`)**:
    * e.g., `./Result/Skipped_Log_internlm__internlm2_5-20b-chat_DIRECT.txt`
    * A text file logging any items that were skipped during processing for that specific combination due to errors (e.g., missing input data, JSON decode errors in input, critical unhandled errors in the pipeline).

## 8. Customization and Extension

* **Adding/Changing Models/Prompts**: Modify `WORKER_MODEL_IDS` and `PROMPT_VERSIONS_TO_TEST` in `settings.json`. If adding new custom prompt versions (e.g., "EXPERT_V2"), ensure you add a corresponding template in `prompts.py` and update `get_worker_prompt_template`.
* **Adjusting Judge Prompts**: The ACCURACY judge prompts (`PROMPT_FOR_JUDGE_L1/L2/L3_ACCURACY_TEMPLATE`) and the `PROMPT_FOR_JUDGE_LLM_TRUE_INTEGRITY_TEMPLATE` in `prompts.py` can be iteratively refined. If you change the JSON output key (e.g., from `is_judged_correct`), update parsing in `llm_calls.py` (`get_accuracy_verdict`).
* **Changing ESI Weights**: Adjust `WEIGHT_...` values in `settings.json`.
* **Modifying Metric Calculations**: Logic within `evaluation_metrics.py` can be updated for any ESI component. For example, to implement a more sophisticated $P_{irr}$ for Efficiency, or a more complex Safety/Alignment score.
* **Concurrency**: Adjust `MAX_CONCURRENT_ITEMS_PER_COMBO` in `settings.json`.

## 9. Troubleshooting

* **`FileNotFoundError` for `settings.json`**: Ensure `settings.json` is in the same directory as `config.py` and `main.py`, and the filename is correct.
* **`JSONDecodeError` for `settings.json`**: Carefully validate `settings.json` syntax (use a JSON linter). Check for missing/extra commas, ensure keys and strings are double-quoted, and numbers are not quoted.
* **API Errors (e.g., `401 Unauthorized`, `429 Too Many Requests`, connection errors)**:
    * Verify all `_API_TOKEN` values in `settings.json` are correct and active.
    * Verify all `_API_URL` values are correct for the specified models.
    * For `429` errors, increase `RETRY_DELAY_SECONDS` and/or decrease `MAX_CONCURRENT_ITEMS_PER_COMBO` in `settings.json`. Check the API provider's rate limit documentation.
* **Low ACC Scores**:
    1.  **Perform Manual Spot-Checks**: This is crucial. Examine items marked incorrect by the ACC judge. Compare `worker_answer_cleaned`, `reference_answer`, `instruction`, and `question`. Read the `accuracy_judge_reasoning`.
    2.  **Iterate on ACC Judge Prompt**: Based on spot-checks, refine the corresponding L1, L2, or L3 ACC judge prompt in `prompts.py` to better align with your desired level of strictness for that dataset type.
    3.  **Check Worker LLM Output**: The Worker LLM might genuinely be performing poorly.
    4.  **Evaluate Reference Answers**: Are your reference answers clear and representative?
    5.  **Experiment with Different Judge LLMs**: Change `ACCURACY_JUDGE_MODEL_ID` (and its URL/Token if needed).
* **`AttributeError` or `KeyError` during Config Loading**: Indicates a mismatch between keys defined as required/expected in `config.py` and what's actually present or correctly formatted in `settings.json`. The error message should point to the problematic key.
* **`tqdm` progress bar issues**: If progress bars are not displaying correctly (e.g., multiple lines), ensure no direct `print()` statements are used inside tight loops managed by `tqdm` in `main.py`'s `process_single_item_full_pipeline`. Use `tqdm.write()` for messages within such loops.

## 10. Future Enhancements (Potential)

* Implement more sophisticated methods for **True Integrity** (e.g., rule-based checks on reasoning steps if CoT is used).
* Develop a more advanced **Alignment** score that goes beyond simple heuristics, potentially using another LLM with specific ethical/value-alignment prompts.
* Add support for different **input data formats** besides JSONL.
* Integrate a **database backend** for storing and querying results.
* Develop a **web interface/dashboard** for easier configuration and visualization of results.
* Implement **model-based calculation of $P_{irr}$** for the Efficiency score.
* Add more **granular error reporting and retry mechanisms** for API calls.