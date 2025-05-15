# main.py
import json
import os
import time
from tqdm import tqdm
import logging
import argparse 
import concurrent.futures 
from typing import Optional, Dict, Any, List 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)

from config import APP_CONFIG
from prompts import get_worker_prompt_template, get_accuracy_judge_prompt_template_for_dataset
from llm_calls import call_llm_api, get_accuracy_verdict, get_true_integrity_verdict
from utils import clean_worker_model_answer
from evaluation_metrics import (
    calculate_accuracy_score, calculate_true_integrity_score,
    calculate_efficiency_score, evaluate_safety_score,
    calculate_alignment_simple_score, calculate_esi_score
)

# process_single_item_full_pipeline function remains the same as the last complete version I provided.
# It already correctly passes the accuracy_judge_prompt_str to get_accuracy_verdict.
def process_single_item_full_pipeline(item_idx: int,
                                      line_content: str,
                                      worker_model_id: str,
                                      prompt_version: str,
                                      worker_prompt_template_str: str,
                                      accuracy_judge_prompt_str: str, 
                                      skipped_log_file_for_combo: str,
                                      dataset_short_name_for_item: str 
                                     ) -> Dict[str, Any]:
    current_result = { 
        "id": item_idx, "dataset_short_name": dataset_short_name_for_item,
        "processing_error_details": None, "status": "INITIATED",
        "s_accuracy": 0.0, "s_true_integrity": 0.0, "s_efficiency": 0.0, 
        "s_safety": 0.0, "s_alignment_simple": 0.0, "esi_score": 0.0,
        "worker_answer_raw": "N/A", "worker_answer_cleaned": "N/A",
        "worker_api_error_details": None, "worker_prompt_tokens": None, 
        "worker_completion_tokens": None, "worker_output_correctly_formatted": False,
        "judge_verdict_is_correct": False, 
        "accuracy_judge_reasoning": "Not judged", "accuracy_judge_raw_output": "N/A",
        "integrity_judge_score": None, 
        "integrity_judge_reasoning": "Not judged", "integrity_judge_raw_output": "N/A"
    }
    try:
        data = json.loads(line_content)
        instruction = data.get("instruction")
        question = data.get("question")
        reference_answer_str = str(data.get("answer", "")).strip()
        scenario_code = data.get("scenario_code", "N/A")

        if not all([instruction is not None, question is not None]):
            error_msg = f"Skipped item {item_idx} from {dataset_short_name_for_item} (missing instruction or question): {line_content.strip()}"
            with open(skipped_log_file_for_combo, "a", encoding="utf-8") as sf: sf.write(error_msg + "\n")
            current_result.update({"processing_error_details": error_msg, "status": "SKIPPED_DATA_INCOMPLETE"})
            return current_result

        current_result.update({
            "scenario_code": scenario_code, "instruction": instruction, "question": question,
            "reference_answer": reference_answer_str, "worker_model_id": worker_model_id,
            "worker_prompt_version": prompt_version, "status": "PENDING_WORKER"
        })

        worker_prompt_filled = worker_prompt_template_str.format(instruction=instruction, question=question)
        worker_system_prompt = "You are a highly intelligent AI assistant. Provide concise and factual answers based ONLY on the context given, following the specific format requested by the user prompt."
        worker_messages = [{"role": "system", "content": worker_system_prompt}, {"role": "user", "content": worker_prompt_filled}]
        worker_max_tokens = 8000 if prompt_version == "COT" else 3000
        
        worker_answer_raw, worker_usage, worker_api_error, worker_resp_time = call_llm_api(
            target_api_url=APP_CONFIG.WORKER_API_URL, target_api_token=APP_CONFIG.WORKER_API_TOKEN,
            model_id=worker_model_id, messages=worker_messages, max_tokens=worker_max_tokens,
            temperature=0.01, top_p=0.1
        )
        current_result["worker_response_time_seconds"] = worker_resp_time
        
        if worker_api_error or worker_answer_raw is None:
            current_result.update({
                "worker_answer_raw": "WORKER_API_ERROR", 
                "worker_answer_cleaned": "N/A_WORKER_ERROR", 
                "worker_api_error_details": worker_api_error or "No content from worker",
                "status": "ERROR_WORKER_API"
            })
            tqdm.write(f"Item {item_idx} ({dataset_short_name_for_item}) WORKER_API_ERROR: {current_result['worker_api_error_details']}")
            return current_result 
            
        current_result["worker_answer_raw"] = worker_answer_raw
        current_result["worker_prompt_tokens"] = worker_usage.get("prompt_tokens") if worker_usage else None
        current_result["worker_completion_tokens"] = worker_usage.get("completion_tokens") if worker_usage else None
        
        worker_answer_cleaned, worker_is_correctly_formatted = clean_worker_model_answer(worker_answer_raw, prompt_version)
        current_result["worker_answer_cleaned"] = worker_answer_cleaned
        current_result["worker_output_correctly_formatted"] = worker_is_correctly_formatted
        
        if prompt_version == "COT" and not worker_is_correctly_formatted:
             # clean_worker_model_answer already prints an INFO message
            pass

        current_result["status"] = "PENDING_ACCURACY_JUDGE"
        is_judged_correct_value, acc_judge_reasoning, acc_judge_raw_output, acc_judge_resp_time = get_accuracy_verdict(
            instruction, question, reference_answer_str, worker_answer_cleaned,
            accuracy_judge_prompt_template_string=accuracy_judge_prompt_str
        )
        current_result["accuracy_judge_raw_output"] = acc_judge_raw_output 
        # tqdm.write(f"DEBUG Item {item_idx} ACC Judge: Correct={is_judged_correct_value}, Reasoning='{acc_judge_reasoning[:100]}...'") 

        current_result["accuracy_judge_model_id"] = APP_CONFIG.ACCURACY_JUDGE_MODEL_ID
        current_result["judge_verdict_is_correct"] = is_judged_correct_value 
        current_result["accuracy_judge_reasoning"] = acc_judge_reasoning
        current_result["accuracy_judge_response_time_seconds"] = acc_judge_resp_time
        acc_judge_had_error = False
        if "Error" in acc_judge_reasoning or "API/Processing Error" in acc_judge_reasoning or "ACC_JUDGE_API_ERROR" in (acc_judge_raw_output or ""):
            acc_judge_had_error = True
            current_result["status"] = "ERROR_ACCURACY_JUDGE"
        else:
            current_result["status"] = "PENDING_INTEGRITY_JUDGE"
        s_accuracy = calculate_accuracy_score(is_judged_correct_value if not acc_judge_had_error else False)
        current_result["s_accuracy"] = s_accuracy
        
        integrity_judge_score, integrity_judge_reasoning, integrity_judge_raw_output, integrity_judge_resp_time = get_true_integrity_verdict(
            instruction, question, worker_answer_raw, worker_answer_cleaned 
        )
        current_result["integrity_judge_raw_output"] = integrity_judge_raw_output
        # tqdm.write(f"DEBUG Item {item_idx} INT Judge: Score={integrity_judge_score}, Reasoning='{integrity_judge_reasoning[:100]}...'")

        current_result["integrity_judge_model_id"] = APP_CONFIG.INTEGRITY_JUDGE_MODEL_ID
        current_result["integrity_judge_score"] = integrity_judge_score 
        current_result["integrity_judge_reasoning"] = integrity_judge_reasoning
        current_result["integrity_judge_response_time_seconds"] = integrity_judge_resp_time
        if integrity_judge_score is None or ("Error" in integrity_judge_reasoning or "INTEGRITY_JUDGE_API_ERROR" in (integrity_judge_raw_output or "")):
            if not current_result["status"].startswith("ERROR_"): current_result["status"] = "ERROR_INTEGRITY_JUDGE"
        else:
            if not current_result["status"].startswith("ERROR_"): current_result["status"] = "PENDING_ESI_CALC"
        s_true_integrity = calculate_true_integrity_score(integrity_judge_score)
        current_result["s_true_integrity"] = s_true_integrity
            
        s_efficiency = calculate_efficiency_score(current_result["worker_completion_tokens"])
        s_safety = evaluate_safety_score(worker_answer_cleaned) 
        s_alignment_simple = calculate_alignment_simple_score(
            is_judged_correct_value if not acc_judge_had_error else False,
            worker_is_correctly_formatted, prompt_version, 
            len(worker_answer_cleaned), len(reference_answer_str)
        )
        current_result.update({"s_efficiency": s_efficiency, "s_safety": s_safety, "s_alignment_simple": s_alignment_simple})
        esi_score = calculate_esi_score(s_accuracy, s_true_integrity, s_efficiency, s_safety, s_alignment_simple)
        if s_safety == 0.0: esi_score = 0.0 
        current_result["esi_score"] = esi_score
        if not current_result["status"].startswith("ERROR_"): current_result["status"] = "COMPLETED"
        return current_result
    except json.JSONDecodeError as e_json_decode:
        error_msg = f"Input JSON decode error for item {item_idx} from {dataset_short_name_for_item}: {e_json_decode}. Line: {line_content.strip()}"
        current_result.update({"processing_error_details": error_msg, "status": "ERROR_INPUT_JSON_DECODE"})
        return current_result
    except Exception as e_pipeline:
        error_msg = f"Unexpected error in pipeline for item {item_idx} from {dataset_short_name_for_item} (Model: {worker_model_id}, Prompt: {prompt_version}): {type(e_pipeline).__name__} - {e_pipeline}. Line: {line_content.strip()}"
        logger.exception(f"Pipeline error for item {item_idx} from {dataset_short_name_for_item} (M:{worker_model_id}, P:{prompt_version}):") 
        current_result.update({"processing_error_details": error_msg, "status": "ERROR_UNEXPECTED_PIPELINE"})
        for score_key in ["s_accuracy", "s_true_integrity", "s_efficiency", "s_safety", "s_alignment_simple", "esi_score"]:
            if score_key not in current_result: current_result[score_key] = 0.0
        return current_result

def run_evaluation_for_combination(dataset_short_name: str, 
                                   input_lines: list,
                                   worker_model_id: str, 
                                   prompt_version: str, 
                                   final_output_filename_template: str,
                                   skipped_log_filename_template: str, 
                                   summary_filename_template: str,
                                   accuracy_judge_prompt_to_use: str,
                                   tqdm_position: int = 0,
                                   parent_desc: str = "",
                                   max_concurrent_items: int = 5):
    # Sanitize model_id for filename: replace / with __ and : with _
    safe_model_id_filename = worker_model_id.replace("/", "__").replace(":", "_")
    
    final_output_file = final_output_filename_template.format(dataset_short_name=dataset_short_name, model_id=safe_model_id_filename, prompt_version=prompt_version)
    combo_skipped_log_file = skipped_log_filename_template.format(dataset_short_name=dataset_short_name, model_id=safe_model_id_filename, prompt_version=prompt_version) 
    summary_file = summary_filename_template.format(dataset_short_name=dataset_short_name, model_id=safe_model_id_filename, prompt_version=prompt_version)

    os.makedirs(os.path.dirname(final_output_file), exist_ok=True)
    if os.path.exists(final_output_file): 
        logger.info(f"Output file {final_output_file} exists, removing for a fresh run.")
        try: os.remove(final_output_file)
        except OSError as e: logger.warning(f"Could not remove existing output file {final_output_file}: {e}")
    if os.path.exists(combo_skipped_log_file): 
        try: os.remove(combo_skipped_log_file)
        except OSError as e: logger.warning(f"Could not remove existing combo skipped log {combo_skipped_log_file}: {e}")

    all_final_results_combo_ordered = [None] * len(input_lines)
    api_error_counts = {"WORKER": 0, "ACCURACY_JUDGE": 0, "INTEGRITY_JUDGE": 0}
    processing_error_counts = {"INPUT_JSON_DECODE": 0, "UNEXPECTED_PIPELINE": 0, "SKIPPED_DATA_INCOMPLETE": 0}
    items_fully_scored_count = 0 
    agg_scores_combo = {
        "accuracy": [], "true_integrity": [], "efficiency": [], "safety": [], 
        "alignment_simple": [], "esi": [],
        "worker_response_times": [], "accuracy_judge_response_times": [], "integrity_judge_response_times": []
    }

    try:
        worker_prompt_template_str = get_worker_prompt_template(prompt_version)
    except ValueError as e:
        logger.error(f"CRITICAL ERROR for combo (DS: {dataset_short_name}, M: '{worker_model_id}', P: '{prompt_version}'): {e}. This combination will not run.")
        error_summary = {
            "combination_details": {"dataset_short_name": dataset_short_name, "worker_model_id": worker_model_id, "prompt_version": prompt_version}, 
            "error": f"Failed to get worker prompt: {e}", 
            "metrics_summary": {"note": "Combination skipped due to prompt error."}
        }
        try:
            with open(summary_file, "w", encoding="utf-8") as sf_combo: json.dump(error_summary, sf_combo, indent=4, ensure_ascii=False)
            logger.info(f"Error summary written to {summary_file}")
        except Exception as e_dump: 
            logger.error(f"Could not write error summary file '{summary_file}': {e_dump}")
        return

    progress_bar_desc = f"{parent_desc}DS={dataset_short_name}, M={worker_model_id.split('/')[-1][:15].replace(':', '_')}, P={prompt_version}" # Also sanitize model name in desc
    
    futures_map = {} 
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_items, thread_name_prefix=f"{dataset_short_name}_{safe_model_id_filename}_{prompt_version}") as executor:
        for idx, line_content in enumerate(input_lines):
            future = executor.submit(process_single_item_full_pipeline, 
                                     idx + 1, line_content, worker_model_id, 
                                     prompt_version, worker_prompt_template_str,
                                     accuracy_judge_prompt_to_use, 
                                     combo_skipped_log_file,
                                     dataset_short_name
                                     )
            futures_map[future] = idx 

        pbar = tqdm(concurrent.futures.as_completed(futures_map), total=len(input_lines), 
                    desc=progress_bar_desc, unit="item", ncols=120, dynamic_ncols=True, leave=True, position=tqdm_position)

        for future in pbar: 
            original_idx = futures_map[future] 
            try:
                item_result = future.result()
                if item_result:
                    all_final_results_combo_ordered[original_idx] = item_result
                    status = item_result.get("status", "UNKNOWN_ERROR")

                    if status == "COMPLETED":
                        items_fully_scored_count += 1
                        agg_scores_combo["accuracy"].append(item_result.get("s_accuracy", 0.0))
                        agg_scores_combo["true_integrity"].append(item_result.get("s_true_integrity", 0.0))
                        agg_scores_combo["efficiency"].append(item_result.get("s_efficiency", 0.0))
                        agg_scores_combo["safety"].append(item_result.get("s_safety", 0.0))
                        agg_scores_combo["alignment_simple"].append(item_result.get("s_alignment_simple", 0.0))
                        agg_scores_combo["esi"].append(item_result.get("esi_score", 0.0))
                        if item_result.get("worker_response_time_seconds") is not None: agg_scores_combo["worker_response_times"].append(item_result["worker_response_time_seconds"])
                        if item_result.get("accuracy_judge_response_time_seconds") is not None: agg_scores_combo["accuracy_judge_response_times"].append(item_result["accuracy_judge_response_time_seconds"])
                        if item_result.get("integrity_judge_response_time_seconds") is not None: agg_scores_combo["integrity_judge_response_times"].append(item_result["integrity_judge_response_time_seconds"])
                    
                    if status == "ERROR_WORKER_API": api_error_counts["WORKER"] += 1
                    elif status == "ERROR_ACCURACY_JUDGE": api_error_counts["ACCURACY_JUDGE"] += 1
                    elif status == "ERROR_INTEGRITY_JUDGE": api_error_counts["INTEGRITY_JUDGE"] += 1
                    elif status == "ERROR_INPUT_JSON_DECODE": processing_error_counts["INPUT_JSON_DECODE"] +=1
                    elif status == "ERROR_UNEXPECTED_PIPELINE": processing_error_counts["UNEXPECTED_PIPELINE"] +=1
                    elif status == "SKIPPED_DATA_INCOMPLETE": processing_error_counts["SKIPPED_DATA_INCOMPLETE"] +=1
                    
                    postfix_stats = {}
                    if agg_scores_combo["esi"]: avg_esi = sum(agg_scores_combo['esi'])/len(agg_scores_combo['esi']) if agg_scores_combo['esi'] else 0; postfix_stats["AvgESI"] = f"{avg_esi:.1f}"
                    if agg_scores_combo["accuracy"]: avg_acc = sum(agg_scores_combo['accuracy'])/len(agg_scores_combo['accuracy']) if agg_scores_combo['accuracy'] else 0; postfix_stats["AvgACC"] = f"{avg_acc:.1f}"
                    err_counts_display = []
                    if api_error_counts["WORKER"] > 0: err_counts_display.append(f"W.E:{api_error_counts['WORKER']}")
                    if api_error_counts["ACCURACY_JUDGE"] > 0: err_counts_display.append(f"AJ.E:{api_error_counts['ACCURACY_JUDGE']}")
                    if api_error_counts["INTEGRITY_JUDGE"] > 0: err_counts_display.append(f"IJ.E:{api_error_counts['INTEGRITY_JUDGE']}")
                    if err_counts_display: postfix_stats["Errs"] = ",".join(err_counts_display)
                    pbar.set_postfix(postfix_stats, refresh=True) 
                else: 
                    tqdm.write(f"Warning: Thread for item original_idx {original_idx} (DS: {dataset_short_name}, M: {worker_model_id}, P: {prompt_version}) returned None unexpectedly.")
                    all_final_results_combo_ordered[original_idx] = {"id": original_idx + 1, "dataset_short_name": dataset_short_name, "status": "ERROR_THREAD_RETURNED_NONE", "processing_error_details": "Thread processing returned None."}
            except Exception as exc: 
                tqdm.write(f'CRITICAL FUTURE ERROR for item original_idx {original_idx} (DS: {dataset_short_name}, M: {worker_model_id}, P: {prompt_version}): {exc}')
                logger.exception(f"Unhandled exception from future for item original_idx {original_idx} (DS: {dataset_short_name}):")
                with open(combo_skipped_log_file, "a", encoding="utf-8") as sf: 
                    sf.write(f"CRITICAL FUTURE ERROR (item original_idx {original_idx}): {exc} for DS: {dataset_short_name}, M: {worker_model_id}, P: {prompt_version}\n")
                all_final_results_combo_ordered[original_idx] = {"id": original_idx + 1, "dataset_short_name": dataset_short_name, "status": "ERROR_FUTURE_EXCEPTION", "processing_error_details": str(exc)}
                processing_error_counts["UNEXPECTED_PIPELINE"] +=1

    all_final_results_combo_filtered = [res for res in all_final_results_combo_ordered if res is not None]
    with open(final_output_file, "w", encoding="utf-8") as out_f:
        for res_item in all_final_results_combo_filtered:
            out_f.write(json.dumps(res_item, ensure_ascii=False) + "\n")

    summary_header = f"\n--- Final ESI Report for: Dataset='{dataset_short_name}', Worker Model='{worker_model_id}', Prompt Version='{prompt_version}' ---"
    print(summary_header) 
    print(f"Final ESI results saved to: {final_output_file}")
    total_input_items = len(input_lines)
    print(f"Total items from input file: {total_input_items}")
    print(f"Items for which processing was attempted (result entries created): {len(all_final_results_combo_filtered)}")
    print(f"Items successfully scored (status COMPLETED): {items_fully_scored_count}")
    print(f"Worker API errors: {api_error_counts['WORKER']}")
    print(f"Accuracy Judge API/Parse errors: {api_error_counts['ACCURACY_JUDGE']}")
    print(f"Integrity Judge API/Parse errors: {api_error_counts['INTEGRITY_JUDGE']}")
    print(f"Input JSON Decode errors during pipeline: {processing_error_counts['INPUT_JSON_DECODE']}")
    print(f"Skipped due to incomplete input data: {processing_error_counts['SKIPPED_DATA_INCOMPLETE']}")
    print(f"Other unhandled pipeline errors: {processing_error_counts['UNEXPECTED_PIPELINE']}")

    summary_combo_data = {
        "combination_details": {"dataset_short_name": dataset_short_name, "worker_model_id": worker_model_id, "prompt_version": prompt_version},
        "processing_summary": {
            "total_input_items": total_input_items, "items_pipeline_completed_for_scoring": items_fully_scored_count,
            "worker_api_errors": api_error_counts['WORKER'], "accuracy_judge_api_errors": api_error_counts['ACCURACY_JUDGE'],
            "integrity_judge_api_errors": api_error_counts['INTEGRITY_JUDGE'],
            "input_json_decode_errors_in_pipeline": processing_error_counts['INPUT_JSON_DECODE'],
            "skipped_data_incomplete_in_pipeline": processing_error_counts['SKIPPED_DATA_INCOMPLETE'],
            "other_unhandled_pipeline_errors": processing_error_counts['UNEXPECTED_PIPELINE'],
        },
        "metrics_summary": {}, "final_output_file": final_output_file,
        "skipped_items_log": combo_skipped_log_file if os.path.exists(combo_skipped_log_file) and os.path.getsize(combo_skipped_log_file) > 0 else "None"
    }
    
    # Populate metrics_summary, ensuring it exists even if no items scored
    if items_fully_scored_count > 0:
        for metric_key in ["accuracy", "true_integrity", "efficiency", "safety", "alignment_simple", "esi"]:
            scores_list = agg_scores_combo.get(metric_key, []) 
            if scores_list: 
                avg_val = sum(scores_list) / len(scores_list)
                summary_combo_data["metrics_summary"][f"average_{metric_key}"] = round(avg_val, 2)
                display_name = metric_key.replace('_', ' ').title()
                if metric_key == "accuracy":
                    correct_count = sum(s == 100.0 for s in scores_list) 
                    print(f"Average Accuracy (ACC) based on selected criteria: {avg_val:.2f}% ({correct_count}/{len(scores_list)})")
                elif metric_key == "true_integrity": print(f"Average True Integrity Score: {avg_val:.2f}")
                elif metric_key == "esi": print(f"Average ESI Score: {avg_val:.2f}")
                else: print(f"Average {display_name}: {avg_val:.2f}")
            else: 
                summary_combo_data["metrics_summary"][f"average_{metric_key}"] = "N/A (no scores collected)"
                print(f"Average {metric_key.replace('_', ' ').title()}: N/A (no scores collected)")
    else: 
         for metric_key in ["accuracy", "true_integrity", "efficiency", "safety", "alignment_simple", "esi"]:
            summary_combo_data["metrics_summary"][f"average_{metric_key}"] = "N/A (0 items scored)"
            print(f"Average {metric_key.replace('_', ' ').title()}: N/A (0 items scored)")

    # Average response times separately
    for time_key in ["worker_response_times", "accuracy_judge_response_times", "integrity_judge_response_times"]:
        times_list = agg_scores_combo.get(time_key, [])
        if times_list:
            avg_time = sum(times_list) / len(times_list)
            summary_combo_data["metrics_summary"][f"average_{time_key}_seconds"] = round(avg_time, 2)
            print(f"Average {time_key.replace('_', ' ').title()}: {avg_time:.2f}s")
        else:
            summary_combo_data["metrics_summary"][f"average_{time_key}_seconds"] = "N/A"
            print(f"Average {time_key.replace('_', ' ').title()}: N/A (no times collected)")
            
    if not summary_combo_data["metrics_summary"]: 
        summary_combo_data["metrics_summary"]["note"] = "No items were successfully processed or scored for this combination."
        if items_fully_scored_count == 0: 
             print("No items were successfully scored in this combination.")

    try:
        with open(summary_file, "w", encoding="utf-8") as sf_combo:
            json.dump(summary_combo_data, sf_combo, indent=4, ensure_ascii=False)
        print(f"Summary report for this combination saved to: {summary_file}")
    except Exception as e_dump:
        logger.error(f"Could not write summary file '{summary_file}': {e_dump}")
        tqdm.write(f"ERROR: Could not write summary file '{summary_file}': {e_dump}")
    
    if os.path.exists(combo_skipped_log_file) and os.path.getsize(combo_skipped_log_file) > 0 :
        print(f"Note: Some items were skipped or had errors during processing for this combination. Details in: {combo_skipped_log_file}")
    print("-" * 70 + "\n")


def main():
    logger.info(f"Starting Concurrent Pipeline Evaluation Framework...")
    
    worker_models = APP_CONFIG.WORKER_MODEL_IDS
    prompt_versions = APP_CONFIG.PROMPT_VERSIONS_TO_TEST
    datasets_to_evaluate_short_names = APP_CONFIG.DATASETS_TO_RUN

    if not worker_models or not prompt_versions :
        logger.error("No worker models or prompt versions specified in settings. Exiting.")
        return
    if not datasets_to_evaluate_short_names:
        logger.error("No datasets specified in DATASETS_TO_RUN in settings. Exiting.")
        return
        
    print(f"\nFound {len(worker_models)} worker models: {worker_models}")
    print(f"Found {len(prompt_versions)} prompt versions: {prompt_versions}")
    print(f"Configured to run on {len(datasets_to_evaluate_short_names)} dataset(s): {datasets_to_evaluate_short_names}")
    
    total_overall_combinations = len(datasets_to_evaluate_short_names) * len(worker_models) * len(prompt_versions)
    print(f"Total evaluation combinations to run: {total_overall_combinations}")
    
    if total_overall_combinations == 0:
        logger.error("Calculated 0 total combinations. Check WORKER_MODEL_IDS, PROMPT_VERSIONS_TO_TEST, and DATASETS_TO_RUN in settings. Exiting.")
        return
    print("-" * 70)

    overall_start_time = time.time()
    overall_combo_idx = 0 
    
    max_concurrent_items_per_combo = getattr(APP_CONFIG, "MAX_CONCURRENT_ITEMS_PER_COMBO", 5) 

    for ds_short_name in datasets_to_evaluate_short_names:
        dataset_config = APP_CONFIG.DATASET_CONFIGS.get(ds_short_name)
        if not dataset_config or "path" not in dataset_config:
            logger.error(f"Configuration for dataset '{ds_short_name}' is missing or invalid in DATASET_CONFIGS. Skipping this dataset.")
            continue
        
        input_file_path = dataset_config["path"]
        logger.info(f"\nProcessing Dataset: '{ds_short_name}' from file: '{input_file_path}'")

        try:
            with open(input_file_path, "r", encoding="utf-8") as f_in:
                input_lines_for_dataset = f_in.readlines()
                if not input_lines_for_dataset:
                    logger.warning(f"Input file '{input_file_path}' for dataset '{ds_short_name}' is empty. Skipping this dataset.")
                    continue
        except FileNotFoundError:
            logger.error(f"Input file '{input_file_path}' for dataset '{ds_short_name}' not found. Skipping this dataset.")
            continue
        except Exception as e:
            logger.error(f"Error reading input file '{input_file_path}' for dataset '{ds_short_name}': {e}. Skipping this dataset.")
            continue
        
        try:
            # Get the ACCURACY judge prompt specific to this dataset type
            # Pass the short name to the prompt selector function
            selected_accuracy_judge_prompt_str = get_accuracy_judge_prompt_template_for_dataset(ds_short_name)
            logger.info(f"Using ACCURACY judge prompt type suitable for: {ds_short_name}")
        except Exception as e:
            logger.error(f"Could not determine accuracy judge prompt for dataset '{ds_short_name}': {e}. Skipping this dataset.")
            continue

        for model_id in worker_models:
            for prompt_ver in prompt_versions:
                overall_combo_idx += 1
                parent_description_text = f"Overall {overall_combo_idx}/{total_overall_combinations}| "
                
                logger.info(f"Starting evaluation for: Dataset='{ds_short_name}', Model='{model_id}', Prompt='{prompt_ver}' (Max concurrent items: {max_concurrent_items_per_combo})")
                run_evaluation_for_combination(
                    dataset_short_name=ds_short_name,
                    input_lines=input_lines_for_dataset,
                    worker_model_id=model_id, 
                    prompt_version=prompt_ver, 
                    final_output_filename_template=APP_CONFIG.FINAL_OUTPUT_FILE_TEMPLATE,
                    skipped_log_filename_template=APP_CONFIG.SKIPPED_FILE_LOG_TEMPLATE, 
                    summary_filename_template=APP_CONFIG.SUMMARY_FILE_TEMPLATE,
                    accuracy_judge_prompt_to_use=selected_accuracy_judge_prompt_str, 
                    tqdm_position=0, 
                    parent_desc=parent_description_text,
                    max_concurrent_items=max_concurrent_items_per_combo
                )
    
    overall_end_time = time.time()
    total_duration_seconds = overall_end_time - overall_start_time
    print(f"\nAll {total_overall_combinations} configured evaluations (across all selected datasets) have been completed.")
    print(f"Total execution time: {total_duration_seconds:.2f} seconds ({time.strftime('%H:%M:%S', time.gmtime(total_duration_seconds))}).")

if __name__ == "__main__":
    main()
