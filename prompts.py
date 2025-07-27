# -*- coding: utf-8 -*-
"""
This file defines prompt templates for interacting with Large Language Models (LLMs).

The system involves two primary roles:
1.  Worker: An LLM that generates answers based on instructions.
2.  Fallback Extractor / Parser: An LLM invoked only when the primary process 
    (regular expression parsing) fails. Its purpose is data cleaning and 
    extraction, not evaluation.
"""
import os

# ==============================================================================
# 1. WORKER PROMPTS (For guiding the Worker LLM to generate answers)
# ==============================================================================

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

# =================================================================================================
# 2. FALLBACK ANSWER EXTRACTION PROMPTS (Used only when regex extraction fails, to guide the model in parsing raw output)
#    Note: This model does not perform final evaluation. It only confirms if the 'Reference Answer' 
#          is present in the 'Candidate's' messy text.
# =================================================================================================

# For L1: (Loose Extraction - Reasonable Inference)
PROMPT_FOR_FALLBACK_EXTRACTOR_L1_TEMPLATE = (
    "You are an AI Data Parser. Your task is to determine if the core information from the 'Reference Answer' can be reasonably inferred from the 'Candidate's Raw Output'. "
    "The Candidate's output is from another AI and may be malformed. Your goal is to check if the intended answer can be salvaged.\n\n"
    "**Evaluation Focus: Plausible Inference**\n"
    "Consider the answer 'salvageable' (`is_judged_correct`: true) if:\n"
    "1. The Raw Output, through a reasonable line of thought, implies the Reference Answer.\n"
    "2. The core meaning is approximately correct, even if details are missing or slightly off.\n"
    "Do not be strict about phrasing. The primary concern is whether a reasonable interpretation of the messy output points to the reference answer.\n\n"
    "Mark as 'not salvageable' (`is_judged_correct`: false) if the output is fundamentally incorrect, irrelevant, or requires unreasonable leaps of logic to connect to the reference answer.\n\n"
    "Here is the information you need to evaluate:\n"
    "1. Instruction (Original Context):\n```\n{instruction}\n```\n\n"
    "2. Question (Original Question):\n```\n{question}\n```\n\n"
    "3. Reference Answer (The ground truth to look for):\n```\n{reference_answer}\n```\n\n"
    "4. Candidate's Raw Output (The messy text to parse):\n```\n{candidate_answer}\n```\n\n"
    "**Output Format:**\n"
    "Respond ONLY with a JSON object with two keys:\n"
    "1. `\"is_judged_correct\"`: A boolean value (true if the reference answer is plausibly present, false otherwise).\n"
    "2. `\"reasoning\"`: A brief explanation for your decision.\n\n"
    "Now, provide your parsing result:"
)

# For L2: (Standard Extraction - Sensibly Correct)
PROMPT_FOR_FALLBACK_EXTRACTOR_L2_TEMPLATE = (
    "You are an AI Data Parser. Your task is to determine if the core information from the 'Reference Answer' is sensibly and directly present within the 'Candidate's Raw Output'. "
    "The Candidate's output is from another AI and may be malformed. Your goal is to check if the intended answer can be salvaged.\n\n"
    "**Evaluation Focus: Direct Presence of Core Information**\n"
    "Consider the answer 'salvageable' (`is_judged_correct`: true) if:\n"
    "1. The Raw Output directly addresses the core question and its essential information aligns with the Reference Answer.\n"
    "2. The answer is not just vaguely related but contains the key facts, names, or values from the Reference Answer.\n\n"
    "Mark as 'not salvageable' (`is_judged_correct`: false) if the core claim is incorrect, the output is irrelevant, or it fails to meaningfully contain the information from the Reference Answer.\n\n"
    "Here is the information you need to evaluate:\n"
    "1. Instruction (Original Context):\n```\n{instruction}\n```\n\n"
    "2. Question (Original Question):\n```\n{question}\n```\n\n"
    "3. Reference Answer (The ground truth to look for):\n```\n{reference_answer}\n```\n\n"
    "4. Candidate's Raw Output (The messy text to parse):\n```\n{candidate_answer}\n```\n\n"
    "**Output Format:**\n"
    "Respond ONLY with a JSON object with two keys:\n"
    "1. `\"is_judged_correct\"`: A boolean value (true if the reference answer is directly present, false otherwise).\n"
    "2. `\"reasoning\"`: A brief explanation for your decision.\n\n"
    "Now, provide your parsing result:"
)

# For L3: (Strict Extraction - Largely Consistent)
PROMPT_FOR_FALLBACK_EXTRACTOR_L3_TEMPLATE = (
    "You are a strict AI Data Parser. Your task is to determine if the information in the 'Reference Answer' is substantially and factually present within the 'Candidate's Raw Output', allowing for only minor formatting errors. "
    "The Candidate's output is from another AI and may be malformed.\n\n"
    "**Evaluation Focus: High-Fidelity Information Match**\n"
    "Consider the answer 'salvageable' (`is_judged_correct`: true) ONLY if:\n"
    "1. The primary facts, figures, and entities from the Reference Answer are explicitly stated in the Raw Output.\n"
    "2. There are no significant factual discrepancies.\n\n"
    "Mark as 'not salvageable' (`is_judged_correct`: false) if key facts are incorrect, omitted, or misrepresented. Do not tolerate significant deviations from the Reference Answer's content.\n\n"
    "Here is the information you need to evaluate:\n"
    "1. Instruction (Original Context):\n```\n{instruction}\n```\n\n"
    "2. Question (Original Question):\n```\n{question}\n```\n\n"
    "3. Reference Answer (The ground truth to look for):\n```\n{reference_answer}\n```\n\n"
    "4. Candidate's Raw Output (The messy text to parse):\n```\n{candidate_answer}\n```\n\n"
    "**Output Format:**\n"
    "Respond ONLY with a JSON object with two keys:\n"
    "1. `\"is_judged_correct\"`: A boolean value (true if the reference answer is verifiably present, false otherwise).\n"
    "2. `\"reasoning\"`: A brief explanation for your decision.\n\n"
    "Now, provide your parsing result:"
)


def get_fallback_extractor_prompt_template(level_name: str) -> str:
    """
    Selects the appropriate FALLBACK EXTRACTOR prompt based on the desired strictness level.
    This is used when regex parsing fails.
    
    Args:
        level_name (str): The desired strictness level. Must be one of 'L1', 'L2', or 'L3'.
        
    Returns:
        str: The corresponding prompt template.
        
    Raises:
        ValueError: If an unknown level_name is provided.
    """
    if level_name == "L1":
        print(f"INFO: Using L1 FALLBACK EXTRACTOR prompt. (Leniency: Reasonable Inference)")
        return PROMPT_FOR_FALLBACK_EXTRACTOR_L1_TEMPLATE
    elif level_name == "L2":
        print(f"INFO: Using L2 FALLBACK EXTRACTOR prompt. (Leniency: Sensibly Correct)")
        return PROMPT_FOR_FALLBACK_EXTRACTOR_L2_TEMPLATE
    elif level_name == "L3":
        print(f"INFO: Using L3 FALLBACK EXTRACTOR prompt. (Leniency: Largely Consistent)")
        return PROMPT_FOR_FALLBACK_EXTRACTOR_L3_TEMPLATE
    else:
        raise ValueError(f"Unknown level_name '{level_name}'. Available: L1, L2, L3.")

# ==============================================================================
# 3. DIAGNOSTIC PROMPTS (For analyzing the integrity of the Worker LLM's thought process)
# ==============================================================================

PROMPT_FOR_INTEGRITY_ANALYZER_TEMPLATE = (
    "You are a meticulous AI Process Analyzer. Your task is to assess the **completeness and logical integrity** of an AI assistant's thought process (if visible) and its final output, based on the provided 'Instruction' and 'Question'.\n\n"
    "**Evaluation Focus: Process Integrity & Completeness of Thought**\n"
    "1.  **Condition Coverage:** Did the AI's output address all critical conditions and constraints?\n"
    "2.  **Question Comprehension:** Does the output demonstrate a full understanding of all parts of the Question?\n"
    "3.  **Information Utilization:** Was all relevant information from the 'Instruction' appropriately used?\n"
    "4.  **Logical Flow:** If reasoning steps are visible, are they logical, coherent, and without significant gaps?\n\n"
    "**Do NOT focus on whether the final answer is correct. Focus on the *process and completeness of consideration*.**\n\n"
    "Here is the information you need to analyze:\n"
    "1. Instruction (Original Context):\n```\n{instruction}\n```\n\n"
    "2. Question (Original Question):\n```\n{question}\n```\n\n"
    "3. Candidate Output (Raw - this may include reasoning steps):\n```\n{candidate_output_raw}\n```\n\n"
    "4. Candidate Answer (Cleaned - the final answer extracted from the raw output):\n```\n{candidate_answer_cleaned}\n```\n\n"
    "**Output Format:**\n"
    "Respond ONLY with a JSON object containing two keys:\n"
    "1. `\"integrity_score\"`: A numerical score from 0 to 100, reflecting the completeness of the thought process.\n"
    "2. `\"integrity_reasoning\"`: A brief explanation for the assigned score.\n\n"
    "Now, provide your analysis:"
)
