[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/mit)

---

</div>

## 🌟 Overview

This project provides a framework for evaluating Large Language Models (LLMs) based on an **ESI** score. It processes datasets of questions and reference answers, queries specified "Worker" LLMs, and then uses "Judge" LLMs to assess the responses across multiple dimensions.

**Key Features**:

-   **Multi-Model & Multi-Prompt Evaluation**: Test various LLMs with different prompting strategies.
-   **Configurable Metrics**:
    -   **Accuracy (ACC)**: Correctness judged by an LLM.
    -   **True Integrity (TI)**: Assesses if the LLM comprehensively used provided information (judged by an LLM).
    -   **Efficiency (EFF)**: Based on token usage.
    -   **Safety (SAF)**: Keyword-based check for harmful content.
    -   **Alignment (ALIGN)**: Simple score considering correctness, formatting, and length.
-   **ESI Score**: A weighted composite score of the above metrics.
-   **Concurrent Processing**: Efficiently evaluates multiple items in parallel.
-   **JSONL & JSON Outputs**: Detailed per-item results and aggregated summaries.

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
