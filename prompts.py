# prompts.py
import os 

PROMPT_DIRECT_ANSWER_TEMPLATE = (
    "Based on the 'Background Information (Instruction)' and 'Specific Question (Question)' below, "
    "provide the most direct and concise answer. "
    "The answer must be strictly derived from the provided information ONLY. "
    "It should be a single word, a short phrase, a specific name, a numerical value, a code snippet, or a status description that directly addresses the core of the question. "
    "Do NOT include any explanations, justifications, prefixes (e.g., 'The answer is:'), suffixes, conversational filler, or any information not explicitly stated in the 'Background Information'. "
    "Output only the precise answer itself.\n\n"
    "Background Information (Instruction):\n{instruction}\n\n"
    "Specific Question (Question):\n{question}\n\n"
    "Answer:"
)

PROMPT_COT_ANSWER_TEMPLATE = (
    "Your task is to answer the 'Specific Question (Question)' based on the 'Background Information (Instruction)'.\n"
    "Follow these steps:\n"
    "1. First, carefully analyze the Instruction and the Question.\n"
    "2. Provide a step-by-step reasoning process that shows how you arrive at the answer. Start this section with 'Reasoning:'.\n"
    "3. After your reasoning, on a new line, provide the final, direct, and concise answer. This final answer MUST be prefixed with 'Final Answer: ' (note the space after the colon).\n"
    "The final answer part should be a single word, a short phrase, a specific name, a numerical value, a code snippet, or a status description, derived ONLY from the 'Background Information'.\n"
    "Do not add any other explanations or text after the 'Final Answer: ' prefix and the answer itself.\n\n"
    "Background Information (Instruction):\n{instruction}\n\n"
    "Specific Question (Question):\n{question}\n\n"
)

PROMPT_EXPERT_ANSWER_TEMPLATE = (
    "Leveraging your expertise in lunar exploration engineering, analyze the 'Background Information (Instruction)' and 'Specific Question (Question)' below. "
    "Provide the most direct, concise, and factually accurate answer based SOLELY on the information presented. "
    "Your response should be a single word, a short phrase, a specific name, a numerical value, a code snippet, or a status description that precisely answers the question. "
    "Do NOT include any explanations, justifications, prefixes (e.g., 'The answer is:'), suffixes, conversational filler, or any information not explicitly stated in the 'Background Information'. "
    "Output only the precise answer itself.\n\n"
    "Background Information (Instruction):\n{instruction}\n\n"
    "Specific Question (Question):\n{question}\n\n"
    "Answer:"
)

def get_worker_prompt_template(prompt_version_name: str) -> str:
    if prompt_version_name == "DIRECT":
        return PROMPT_DIRECT_ANSWER_TEMPLATE
    elif prompt_version_name == "COT":
        return PROMPT_COT_ANSWER_TEMPLATE
    elif prompt_version_name == "EXPERT":
        return PROMPT_EXPERT_ANSWER_TEMPLATE
    else:
        raise ValueError(f"Unknown PROMPT_VERSION '{prompt_version_name}'. Available: DIRECT, COT, EXPERT.")

def get_accuracy_judge_prompt_template_for_dataset(dataset_short_name: str) -> str:
    """Selects the appropriate ACCURACY judge prompt based on the dataset short name."""
    name_lower = dataset_short_name.lower() 
    if "l1" in name_lower: 
        print(f"INFO: Using L1 ACCURACY judge prompt for dataset '{dataset_short_name}'. (Reasonable Path, Approx. Correct Result)")
        return PROMPT_FOR_JUDGE_L1_ACCURACY_TEMPLATE
    elif "l2" in name_lower:
        print(f"INFO: Using L2 ACCURACY judge prompt for dataset '{dataset_short_name}'. (Sensibly Correct, More than Just Related)")
        return PROMPT_FOR_JUDGE_L2_ACCURACY_TEMPLATE
    elif "l3" in name_lower:
        print(f"INFO: Using L3 ACCURACY judge prompt for dataset '{dataset_short_name}'. (Largely Consistent Result)")
        return PROMPT_FOR_JUDGE_L3_ACCURACY_TEMPLATE
    else:
        # Default to L2 (balanced) if no specific L1/L3 match for the short name (e.g. for "Demo")
        print(f"WARNING: Dataset short name '{dataset_short_name}' does not match L1, L2, or L3 patterns. Defaulting to L2 ACCURACY judge prompt.")
        return PROMPT_FOR_JUDGE_L2_ACCURACY_TEMPLATE 

PROMPT_FOR_JUDGE_LLM_TRUE_INTEGRITY_TEMPLATE = (
    "You are a meticulous AI Evaluator focusing on the **completeness and logical integrity** of an AI assistant's thought process and output. Your task is to assess if the AI assistant has thoroughly considered all relevant aspects of the 'Instruction' and 'Question' to arrive at its 'Candidate Output'.\n\n"
    "**Evaluation Focus: Process Integrity & Completeness**\n"
    "1.  **Condition Coverage:** Did the AI's output (including any reasoning steps if visible) explicitly or implicitly address all critical conditions, constraints, and pieces of information provided in the 'Instruction' and 'Question'?\n"
    "2.  **Question Comprehension:** Does the output demonstrate a full understanding of all parts of the 'Question'? Or does it only focus on a subset?\n"
    "3.  **Information Utilization:** Was relevant information from the 'Instruction' appropriately used in the reasoning or to formulate the answer? Were any crucial pieces of provided information ignored?\n"
    "4.  **Logical Flow (if reasoning is present):** If a thought process or reasoning steps are visible in the 'Candidate Output (Raw)', are these steps logical, coherent, and without significant gaps or unjustified leaps?\n"
    "5.  **Completeness of Answer relative to Question:** Even if the final cleaned answer is brief, does the overall output (raw or cleaned, as appropriate) suggest that the AI *considered* what was necessary to answer comprehensively?\n\n"
    "**Do NOT focus on whether the final answer is correct in this evaluation (that is handled separately). Focus on the *process and completeness of consideration*.**\n\n"
    "Here is the information you need to evaluate:\n"
    "1. Instruction (Original Context):\n```\n{instruction}\n```\n\n"
    "2. Question (Original Question):\n```\n{question}\n```\n\n"
    "3. Candidate Output (Raw - this may include reasoning steps if provided by the worker AI):\n```\n{candidate_output_raw}\n```\n\n"
    "4. Candidate Answer (Cleaned - the final concise answer extracted from the raw output):\n```\n{candidate_answer_cleaned}\n```\n\n"
    "**Output Format:**\n"
    "Please respond **ONLY** with a JSON object containing two keys:\n"
    "1. `\"integrity_score\"`: A numerical score from 0 to 100, where:\n"
    "   - 0-30: Severely lacking integrity; missed most critical conditions or showed flawed reasoning.\n"
    "   - 31-60: Partially addressed conditions/question parts; some notable omissions or minor logical gaps.\n"
    "   - 61-90: Mostly complete; addressed most key aspects well with minor room for improvement in thoroughness.\n"
    "   - 91-100: Excellent integrity; comprehensively considered all relevant conditions and parts of the question with clear, sound reasoning (if visible).\n"
    "2. `\"integrity_reasoning\"`: A brief explanation for the assigned integrity_score, highlighting key observations about condition coverage, information use, or logical flow.\n\n"
    "Example JSON response:\n"
    "{{\n"
    "  \"integrity_score\": 85,\n"
    "  \"integrity_reasoning\": \"The AI considered most conditions from the instruction but did not explicitly address the time constraint mentioned in the question.\"\n"
    "}}\n\n"
    "Now, provide your evaluation for the given data in the specified JSON format only:"
)
