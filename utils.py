# utils.py
import re

def clean_worker_model_answer(raw_answer_text: str, prompt_version: str) -> tuple[str, bool]:
    """
    Cleans the raw answer text from the worker LLM.
    Returns:
        - cleaned_answer (str)
        - is_correctly_formatted_output (bool): True if CoT format was followed (if CoT was used), True for other prompt types by default.
    """
    answer = raw_answer_text.strip()
    is_correctly_formatted_output = True 

    if prompt_version == "COT":
        match = re.search(r"Final Answer:\s*(.*)", answer, re.IGNORECASE | re.DOTALL)
        if match:
            answer = match.group(1).strip()
        else:
            is_correctly_formatted_output = False 
            # Using print for this specific info as it's about an expected format from the worker
            print(f"\nINFO (Worker Output): COT prompt used, but 'Final Answer:' marker not found. Cleaning applied to full raw output. Raw Preview: \"{raw_answer_text[:100]}...\"")
    
    prefixes_to_remove = [
        "Answer is:", "Answer:", "The answer is:", "The final answer is ", "Expert Answer:",
        "答案是：", "答案：", "答案是", "答案", "好的，答案是：", "好的，答案是", "了解，答案是：", "了解，答案是" # Retained for robustness
    ] 
    
    temp_answer = answer 
    for prefix in prefixes_to_remove:
        if temp_answer.lower().startswith(prefix.lower()):
            temp_answer = temp_answer[len(prefix):].strip()
            break 
    answer = temp_answer 

    if answer.startswith("- "): answer = answer[2:].strip()
    if answer.startswith("* "): answer = answer[2:].strip()
    
    if len(answer) > 1 and ((answer.startswith('"') and answer.endswith('"')) or \
                           (answer.startswith("'") and answer.endswith("'"))):
        answer = answer[1:-1]

    if len(answer) > 1 and (answer.endswith(".") or answer.endswith("。")): # Handles English and Chinese periods
         answer = answer[:-1].strip()
            
    if answer.startswith("> "): answer = answer[2:].strip()
    if answer.startswith(">"): answer = answer[1:].strip()
        
    return answer.strip(), is_correctly_formatted_output