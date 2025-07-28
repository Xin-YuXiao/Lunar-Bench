[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/mit)

[Update] Updated naming and logic errors in Prompt.py in 7/27 4.21 P.M.

## 🌟 Overview

**Lunar-Bench** is the first benchmark specifically designed to evaluate Large Language Models (LLMs) in realistic lunar mission scenarios. Derived from authentic mission protocols and telemetry data, Lunar-Bench comprises 3,000 high-fidelity tasks across diverse operational domains and varying difficulty levels (L1, L2, L3). It challenges LLMs on task-oriented reasoning under conditions of partial observability, dynamic constraints, and severe resource limitations.

**Key Features**:

![image](https://github.com/user-attachments/assets/6bb25c7c-f428-41ef-97d2-26dc291ebda6)

## 📊 ESI Metric Framework

To move beyond conventional task-level accuracy, the **Environmental Scenario Indicators (ESI)** provide a structured, multi-faceted framework for quantifying the nuanced qualities of LLM reasoning within mission-critical lunar contexts. While standard Accuracy captures final correctness, ESI is designed to dissect how models reason, plan, and interact.

![image](https://github.com/user-attachments/assets/a6be27bc-e01a-4e2e-b0cb-0c6b50aed81f)


## 🚀 How to Use

### 1. Prerequisites

-   Python (3.8+ recommended).
-   Install dependencies:
    ```bash
    pip install requests tqdm
    ```

### 2. Setup & Configuration

1.  **Clone/Download Project**: Obtain all project files (`main.py`, `config.py`, `settings.json`, etc.).
2.  **Directory Structure**:
    ```
    .
    ├── Data Demo/              # Your .jsonl datasets
    │   ├── L1-1K.jsonl
    │   └── ...
    ├── Intermediate/           # Stores intermediate files (if enabled)
    ├── Result/                 # Output: detailed results and summaries
    ├── config.py
    ├── evaluation_metrics.py
    ├── llm_calls.py
    ├── main.py                 # Main script to run
    ├── prompts.py
    ├── settings.json           # CRITICAL: Configure this file
    └── utils.py
    ```
3.  **Configure `settings.json`**: This is the **most important step**.
    * **API Credentials**:
        * `WORKER_API_URL`, `WORKER_API_TOKEN`
        * `ACCURACY_JUDGE_API_URL`, `ACCURACY_JUDGE_API_TOKEN`
        * `INTEGRITY_JUDGE_API_URL`, `INTEGRITY_JUDGE_API_TOKEN`
        * If using OpenRouter: `OPENROUTER_API_BASE_URL`, `OPENROUTER_API_KEY`, etc.
        * **Security**: Avoid committing real API keys. Consider environment variables for production/shared use.
    * **Models**:
        * `WORKER_MODEL_IDS`: List of worker LLM IDs to test (e.g., `["openai/gpt-4o", "meta-llama/Llama-3-8b-chat-hf"]`).
        * `ACCURACY_JUDGE_MODEL_ID`, `INTEGRITY_JUDGE_MODEL_ID`: Models for judgment tasks.
    * **Datasets**:
        * `DATASET_CONFIGS`: Define your datasets. Each entry maps a short name (e.g., `"L1"`) to an object with a `"path"` (e.g., `"./Data Demo/L1-1K.jsonl"`) and `"description"`.
        * Dataset files must be in **`.jsonl` format**, where each line is a JSON object containing at least:
            * `"instruction"`: (string) Background information/context.
            * `"question"`: (string) The question for the LLM.
            * `"answer"`: (string) The reference/ground truth answer.
        * `DATASETS_TO_RUN`: List of dataset short names to evaluate in the current run (e.g., `["L1", "L2"]`).
    * **Prompts**:
        * `PROMPT_VERSIONS_TO_TEST`: List of prompt strategies (e.g., `["DIRECT", "COT"]`). These correspond to templates in `prompts.py`.
    * **Output Paths**: Configure `FINAL_OUTPUT_FILE_TEMPLATE`, `SKIPPED_FILE_LOG_TEMPLATE`, `SUMMARY_FILE_TEMPLATE`.
    * **Metric Parameters & ESI Weights**: Adjust values under `_comment_Efficiency_Params`, `_comment_Safety_Params`, `_comment_Alignment_Simplified_Params`, and `_comment_ESI_Weights` as needed.
    * **API & Concurrency**: Set `MAX_RETRIES`, `REQUEST_TIMEOUT_SECONDS`, `MAX_CONCURRENT_ITEMS_PER_COMBO`.

### 4. Prepare Datasets

-   Create your `.jsonl` dataset files according to the format specified above.
-   Place them in the relevant directory (e.g., `Data Demo/`) and ensure paths in `settings.json` are correct.

### 5. Run Evaluation

Execute the main script from the project's root directory:

```bash
python main.py
