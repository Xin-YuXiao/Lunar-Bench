# llm_calls.py
import requests
import time
import json
import re
from typing import Tuple, Optional, Dict, Any
from config import APP_CONFIG 
from prompts import PROMPT_FOR_JUDGE_LLM_TRUE_INTEGRITY_TEMPLATE # Only Integrity prompt needed here directly

def call_llm_api(target_api_url: str, 
                 target_api_token: str, 
                 model_id: str,
                 messages: list,
                 max_tokens: int,
                 temperature: float,
                 top_p: float) -> Tuple[Optional[str], Optional[Dict[str, int]], Optional[str], Optional[float]]:
    payload = {
        "model": model_id, "messages": messages, "max_tokens": max_tokens,
        "temperature": temperature, "top_p": top_p, "stream": False 
    }
    headers = {"Authorization": f"Bearer {target_api_token}", "Content-Type": "application/json"}
    
    if "openrouter.ai" in target_api_url: # Add OpenRouter specific headers
        if hasattr(APP_CONFIG, 'OPENROUTER_HTTP_REFERER') and APP_CONFIG.OPENROUTER_HTTP_REFERER:
            headers["HTTP-Referer"] = APP_CONFIG.OPENROUTER_HTTP_REFERER
        if hasattr(APP_CONFIG, 'OPENROUTER_X_TITLE') and APP_CONFIG.OPENROUTER_X_TITLE:
            headers["X-Title"] = APP_CONFIG.OPENROUTER_X_TITLE

    raw_response_content_for_error = ""
    start_time = time.time(); response_time_seconds = None 
    for attempt in range(APP_CONFIG.MAX_RETRIES): 
        response_obj = None 
        try:
            response_obj = requests.post(target_api_url, headers=headers, json=payload, timeout=APP_CONFIG.REQUEST_TIMEOUT_SECONDS) 
            response_time_seconds = time.time() - start_time
            response_obj.raise_for_status()
            response_data = response_obj.json()
            choices = response_data.get("choices")
            if choices and len(choices) > 0:
                message_obj = choices[0].get("message") 
                if not message_obj and "delta" in choices[0]: message_obj = choices[0].get("delta")
                if message_obj:
                    content = message_obj.get("content", "")
                    usage_data = response_data.get("usage")
                    return content, usage_data, None, response_time_seconds
            error_msg = f"API response from {model_id} at {target_api_url} lacked expected content."
            print(f"\nAPI_CALL_ERROR: {error_msg} (Attempt {attempt+1}/{APP_CONFIG.MAX_RETRIES}) Response: {response_data}")
            raw_response_content_for_error = f"LLM_RESPONSE_STRUCTURE_ERROR: {response_data}"
            if attempt < APP_CONFIG.MAX_RETRIES - 1: time.sleep(APP_CONFIG.RETRY_DELAY_SECONDS * (attempt + 1))
            else: return None, None, raw_response_content_for_error, response_time_seconds
        except requests.exceptions.RequestException as e:
            if response_time_seconds is None: response_time_seconds = time.time() - start_time
            error_msg = f"API Request to {model_id} at {target_api_url} Failed (Attempt {attempt+1}/{APP_CONFIG.MAX_RETRIES}): {type(e).__name__} - {e}"
            print(f"\nAPI_CALL_ERROR: {error_msg}")
            raw_response_content_for_error = f"LLM_API_REQUEST_ERROR: {e}"
            if attempt < APP_CONFIG.MAX_RETRIES - 1: time.sleep(APP_CONFIG.RETRY_DELAY_SECONDS * (attempt + 1))
            else: return None, None, raw_response_content_for_error, response_time_seconds
        except json.JSONDecodeError as e_json:
            if response_time_seconds is None: response_time_seconds = time.time() - start_time
            resp_text = response_obj.text if response_obj else "N/A"
            error_msg = f"Error decoding API JSON from {model_id} at {target_api_url} (Attempt {attempt+1}/{APP_CONFIG.MAX_RETRIES}): {e_json}. Text: {resp_text[:500]}"
            print(f"\nAPI_CALL_ERROR: {error_msg}")
            raw_response_content_for_error = f"LLM_JSON_DECODE_ERROR: {e_json}. Raw: {resp_text[:500]}"
            if attempt < APP_CONFIG.MAX_RETRIES - 1: time.sleep(APP_CONFIG.RETRY_DELAY_SECONDS * (attempt + 1))
            else: return None, None, raw_response_content_for_error, response_time_seconds
        except Exception as e_inner:
            if response_time_seconds is None: response_time_seconds = time.time() - start_time
            resp_text = response_obj.text if response_obj and hasattr(response_obj, 'text') else "N/A"
            error_msg = f"Unexpected error processing API response from {model_id} at {target_api_url} (Attempt {attempt+1}/{APP_CONFIG.MAX_RETRIES}): {type(e_inner).__name__} - {e_inner}. Text: {resp_text[:200]}"
            print(f"\nAPI_CALL_ERROR: {error_msg}")
            raw_response_content_for_error = f"LLM_UNEXPECTED_PROCESSING_ERROR: {e_inner}. Raw: {resp_text[:200]}"
            if attempt < APP_CONFIG.MAX_RETRIES - 1: time.sleep(APP_CONFIG.RETRY_DELAY_SECONDS * (attempt + 1))
            else: return None, None, raw_response_content_for_error, response_time_seconds
    return None, None, f"Max retries reached for {model_id} at {target_api_url}.", response_time_seconds

def get_accuracy_verdict(instruction: str, question: str, 
                         reference_answer: str, candidate_answer: str,
                         accuracy_judge_prompt_template_string: str) -> Tuple[bool, str, str, Optional[float]]:
    judge_prompt_filled = accuracy_judge_prompt_template_string.format(
        instruction=instruction, question=question,
        reference_answer=reference_answer, candidate_answer=candidate_answer
    )
    judge_system_prompt = "You are an expert AI evaluator for accuracy. Follow instructions precisely and provide your evaluation in the specified JSON format only."
    judge_messages = [{"role": "system", "content": judge_system_prompt}, {"role": "user", "content": judge_prompt_filled}]

    judge_response_text, _, judge_api_error, judge_response_time = call_llm_api(
        target_api_url=APP_CONFIG.ACCURACY_JUDGE_API_URL,     
        target_api_token=APP_CONFIG.ACCURACY_JUDGE_API_TOKEN, 
        model_id=APP_CONFIG.ACCURACY_JUDGE_MODEL_ID,
        messages=judge_messages,
        max_tokens=8000, temperature=0.0, top_p=0.1
    )
    
    default_reasoning_on_error = "Accuracy Judge call failed or returned malformed data."
    if judge_api_error or not judge_response_text or judge_response_text.startswith("LLM_"):
        err_msg = f"Accuracy Judge LLM API/Processing Error: {judge_response_text or judge_api_error}"
        print(f"\nJUDGE_ERROR (ACC): {err_msg}")
        return False, err_msg, judge_response_text or "ACC_JUDGE_API_ERROR", judge_response_time
    try:
        match = re.search(r'\{\s*"is_judged_correct"\s*:\s*(true|false)\s*,\s*"reasoning"\s*:\s*".*?"\s*\}', judge_response_text, re.DOTALL | re.IGNORECASE) 
        if match:
            json_str = match.group(0)
            judge_verdict_json = json.loads(json_str)
            is_judged_correct_value = judge_verdict_json.get("is_judged_correct") 
            reasoning = judge_verdict_json.get("reasoning", "No reasoning provided by accuracy judge.")
            if not isinstance(is_judged_correct_value, bool):
                error_reason = f"Accuracy Judge LLM returned non-boolean for is_judged_correct: '{is_judged_correct_value}'."
                print(f"\nJUDGE_ERROR (ACC): {error_reason}")
                return False, error_reason, judge_response_text, judge_response_time
            return is_judged_correct_value, reasoning, judge_response_text, judge_response_time
        else:
            error_reason = f"Accuracy Judge LLM did not return valid JSON with 'is_judged_correct'. Raw: '{judge_response_text[:300]}...'"
            print(f"\nJUDGE_ERROR (ACC): {error_reason}")
            return False, error_reason, judge_response_text, judge_response_time
    except Exception as e: 
        error_reason = f"Error parsing Accuracy Judge LLM response: {e}. Raw: '{judge_response_text[:300]}...'"
        print(f"\nJUDGE_ERROR (ACC): {error_reason}")
        return False, error_reason, judge_response_text, judge_response_time

def get_true_integrity_verdict(instruction: str, question: str, candidate_output_raw: str, candidate_answer_cleaned: str) -> Tuple[Optional[int], str, str, Optional[float]]:
    integrity_judge_prompt_filled = PROMPT_FOR_JUDGE_LLM_TRUE_INTEGRITY_TEMPLATE.format(
        instruction=instruction, question=question,
        candidate_output_raw=candidate_output_raw, candidate_answer_cleaned=candidate_answer_cleaned
    )
    integrity_judge_system_prompt = "You are an expert AI evaluator for process integrity. Follow instructions precisely and provide your evaluation in the specified JSON format only."
    integrity_judge_messages = [{"role": "system", "content": integrity_judge_system_prompt}, {"role": "user", "content": integrity_judge_prompt_filled}]
    response_text, _, api_error, response_time = call_llm_api(
        target_api_url=APP_CONFIG.INTEGRITY_JUDGE_API_URL,     
        target_api_token=APP_CONFIG.INTEGRITY_JUDGE_API_TOKEN, 
        model_id=APP_CONFIG.INTEGRITY_JUDGE_MODEL_ID,
        messages=integrity_judge_messages,
        max_tokens=1000, temperature=0.0, top_p=0.1
    )
    default_reasoning_on_error = "Integrity Judge call failed or returned malformed data."
    if api_error or not response_text or response_text.startswith("LLM_"):
        err_msg = f"Integrity Judge LLM API/Processing Error: {response_text or api_error}"
        print(f"\nJUDGE_ERROR (INT): {err_msg}")
        return None, err_msg, response_text or "INTEGRITY_JUDGE_API_ERROR", response_time
    try:
        match = re.search(r'\{\s*"integrity_score"\s*:\s*(\d+)\s*,\s*"integrity_reasoning"\s*:\s*".*?"\s*\}', response_text, re.DOTALL | re.IGNORECASE)
        if match:
            json_str = match.group(0)
            verdict_json = json.loads(json_str)
            integrity_score_value_str = match.group(1) 
            integrity_score_value = int(integrity_score_value_str)
            reasoning = verdict_json.get("integrity_reasoning", "No reasoning provided by integrity judge.")
            if not (0 <= integrity_score_value <= 100):
                error_reason = f"Integrity Judge LLM returned invalid integrity_score: '{integrity_score_value}'. Must be int 0-100."
                print(f"\nJUDGE_ERROR (INT): {error_reason}")
                return None, error_reason, response_text, response_time
            return integrity_score_value, reasoning, response_text, response_time
        else:
            error_reason = f"Integrity Judge LLM did not return valid JSON for integrity. Raw: '{response_text[:300]}...'"
            print(f"\nJUDGE_ERROR (INT): {error_reason}")
            return None, error_reason, response_text, response_time
    except Exception as e: 
        error_reason = f"Error parsing Integrity Judge LLM response: {e}. Raw: '{response_text[:300]}...'"
        print(f"\nJUDGE_ERROR (INT): {error_reason}")
        return None, error_reason, response_text, response_time