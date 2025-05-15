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

# --- ACCURACY JUDGE PROMPTS (Context-Dependent) ---

# For L1:  (Reasonable Inference Path, Result Approximately Correct)
PROMPT_FOR_JUDGE_L1_ACCURACY_TEMPLATE = (
    "You are an AI Evaluator. Your task is to determine if the 'Candidate Answer' is **approximately correct and seems to follow a reasonable line of thought (if a thought process is apparent)** in response to the 'Question', considering the 'Instruction'. Minor inaccuracies or stylistic differences from the 'Reference Answer' are acceptable if the core understanding and the final conclusion are generally sound and not fundamentally flawed.\n\n"
    "**Evaluation Focus: Approximate Correctness & Plausibility of Approach**\n"
    "The Candidate Answer does NOT need to be a perfect match to the Reference Answer. The Reference is just one example of a correct answer. The answer should demonstrate a basic, reasonable attempt to address the question based on the provided context.\n\n"
    "Consider the Candidate Answer **'is_judged_correct': true** if:\n"
    "1.  **Addresses Core Question:** The answer makes a clear attempt to address the main point or intent of the Question.\n"
    "2.  **Reasonable (Implicit or Explicit) Path:** \n"
    "    - If reasoning steps are provided (e.g., in a CoT prompt's raw output, which you might infer from the candidate answer if it's complex), these steps should generally make sense and not contain glaring logical fallacies that completely invalidate the answer.\n"
    "    - For direct answers, the answer itself should appear as a plausible or logical outcome derived from a reasonable interpretation of the instruction and question. It shouldn't feel random or completely disconnected from a sensible thought process.\n"
    "3.  **Result Approximately Correct:** The key factual elements or the main conclusion of the Candidate Answer are mostly correct, or at least not fundamentally wrong or severely misleading in the context of the Instruction. Small errors on peripheral details can be tolerated if the main thrust of the answer is acceptable and points in the right direction.\n"
    "4.  **Not Wildly Off-Topic or Critically Misleading:** The answer is relevant to the question and does not introduce information that is critically misleading or makes the answer dangerous.\n\n"
    "Mark as **'is_judged_correct': false** if:\n"
    "- The answer is based on a completely flawed understanding or an unreasonable/illogical thought process (if discernible).\n"
    "- The final result/conclusion is fundamentally incorrect or significantly misleading regarding the core of the Question.\n"
    "- The answer is entirely irrelevant to the question or provides no substantive response to its core.\n"
    "- The answer misses the core point of the question by a very wide margin, indicating a lack of basic comprehension or reasonable inference.\n\n"
    "**Be lenient on phrasing, minor omissions, and slight factual inaccuracies on non-critical details. The primary concern is whether a reasonable thought process (even if simple for direct answers) could lead to an answer that is approximately correct in its main assertion.** It should be more than just 'touching a keyword' but doesn't need to be perfect.\n\n"
    "Here is the information you need to evaluate:\n"
    "1. Instruction (Original Context):\n```\n{instruction}\n```\n\n"
    "2. Question (Original Question):\n```\n{question}\n```\n\n"
    "3. Reference Answer (One example of a correct answer; do not require strict matching. Use it as a guide for what a good answer *could* look like.):\n```\n{reference_answer}\n```\n\n"
    "4. Candidate Answer (The AI's response to evaluate):\n```\n{candidate_answer}\n```\n\n"
    "**Output Format:**\n"
    "Please respond **ONLY** with a JSON object containing two keys:\n"
    "1. `\"is_judged_correct\"`: A boolean value (true if the Candidate Answer is approximately correct and based on a reasonable approach, false otherwise).\n"
    "2. `\"reasoning\"`: A brief explanation for your decision, commenting on the plausibility of the approach and the approximate correctness of the result.\n\n"
    "Now, provide your evaluation based on these L1 criteria (reasonable inference path, approximately correct result):"
)

# For L2: (Relatively Lenient - Sensibly Correct, More than Just Vaguely Related)
PROMPT_FOR_JUDGE_L2_ACCURACY_TEMPLATE = (
    "You are an AI Evaluator with a **relatively lenient but fair** stance. Your task is to determine if the 'Candidate Answer' is **sensibly correct and directly relevant** to the 'Question', based on the 'Instruction' and 'Reference Answer'. The answer must be more than just vaguely related; it should be a valid, even if imperfect or incomplete, response to the question's main intent. Do not require perfect alignment with the 'Reference Answer'.\n\n"
    "**Evaluation Focus: Sensible Correctness & Direct Relevance to Core Question**\n\n"
    "Consider the Candidate Answer **'is_judged_correct': true** if:\n"
    "1.  **Directly Addresses Core Question:** The answer clearly makes a direct attempt to respond to the primary query or intent of the 'Question'.\n"
    "2.  **Core Information is Sound:** The essential factual information or the main conclusion presented in the answer, as it relates to the core question, is correct and consistent with the 'Instruction'.\n"
    "3.  **Relevant and On-Topic:** The answer is primarily focused on the question and does not consist mostly of irrelevant information. It shouldn't just touch a keyword and then go off-topic.\n"
    "4.  **No Significant Misinformation on Core Topic:** The answer does not contain major factual errors or severely misleading statements that would invalidate its usefulness as an answer to the Question's core.\n"
    "5.  Omissions of some details or stylistic differences from the 'Reference Answer' are acceptable if the core of the Candidate Answer is sound and relevant to the question's main point.\n\n"
    "Mark as **'is_judged_correct': false** if:\n"
    "- The answer is largely irrelevant or only tangentially related to the Question's core intent (i.e., 'just touching upon a keyword' without substantively addressing the question is not enough).\n"
    "- The core factual claim of the answer, essential to the question, is clearly incorrect or contradicts the 'Instruction'.\n"
    "- The answer, while possibly containing some correct elements, is overall misleading or fails to address the question's primary intent in a meaningful way.\n\n"
    "**The answer needs to be a reasonable and direct attempt at answering the question's main point correctly, not just a fleeting mention of related terms. Give the benefit of the doubt for minor issues if the core is addressed.**\n\n"
    "Here is the information you need to evaluate:\n"
    "1. Instruction (Original Context):\n```\n{instruction}\n```\n\n"
    "2. Question (Original Question):\n```\n{question}\n```\n\n"
    "3. Reference Answer (A guide to a correct answer; strict matching is not required. Focus on the candidate's response to the question based on the instruction.):\n```\n{reference_answer}\n```\n\n"
    "4. Candidate Answer (The AI's response to evaluate):\n```\n{candidate_answer}\n```\n\n"
    "**Output Format:**\n"
    "Please respond **ONLY** with a JSON object containing two keys:\n"
    "1. `\"is_judged_correct\"`: A boolean value (true if the Candidate Answer is sensibly correct and directly relevant by these relatively lenient standards, false otherwise).\n"
    "2. `\"reasoning\"`: A brief explanation for your decision.\n\n"
    "Now, provide your evaluation based on these L2 criteria (sensibly correct, directly relevant, relatively lenient):"
)

# For L3: (Relatively Strict - Result Largely Consistent)
PROMPT_FOR_JUDGE_L3_ACCURACY_TEMPLATE = (
    "You are a discerning AI Evaluator. Your task is to determine if the 'Candidate Answer' is **largely consistent with the expected correct information and factually sound**, accurately addressing the 'Question' based on the 'Instruction'. While not requiring a verbatim match to the 'Reference Answer', a significant overlap in key facts and meaning is expected. The answer should be substantially correct.\n\n"
    "**Evaluation Focus: Substantial Factual Alignment, Core Consistency, and Accuracy**\n\n"
    "Consider the Candidate Answer **'is_judged_correct': true** if:\n"
    "1.  **Core Factual Alignment:** The primary facts, figures, entities, or conclusions presented in the Candidate Answer align substantially and correctly with those derivable from the 'Instruction' and reflected as correct by the 'Reference Answer' (if reliable).\n"
    "2.  **Addresses Key Aspects of Question Accurately:** The answer covers the most important aspects of the 'Question' with factual accuracy.\n"
    "3.  **No Major Factual Discrepancies or Logical Flaws:** There are no significant factual contradictions with the 'Instruction'. If reasoning is implied or shown, it should be sound.\n"
    "4.  **Meaningful Overlap with Correct Information:** The Candidate Answer, in its essence, conveys a meaning and set of critical information that is largely the same as a known correct answer. Minor differences in detail or phrasing are acceptable if they don't alter the core factual correctness.\n\n"
    "Consider the Candidate Answer **'is_judged_correct': false** if:\n"
    "- It presents key facts or conclusions that are demonstrably incorrect or significantly different from the truth established by the 'Instruction' or a reliable 'Reference Answer'.\n"
    "- It omits a majority of the critical information required to correctly and substantially answer the Question.\n"
    "- It provides an answer that, while possibly touching on the topic, fundamentally misses or misrepresents the core information expected for a largely correct answer.\n"
    "- It contains logical flaws that undermine the validity of its conclusion regarding the question.\n\n"
    "**The emphasis is on whether the candidate answer captures the bulk of the correct information accurately and is factually sound.**\n\n"
    "Here is the information you need to evaluate:\n"
    "1. Instruction (Original Context):\n```\n{instruction}\n```\n\n"
    "2. Question (Original Question):\n```\n{question}\n```\n\n"
    "3. Reference Answer (A strong benchmark for a correct and comprehensive answer. Candidate should largely align with its core facts.):\n```\n{reference_answer}\n```\n\n"
    "4. Candidate Answer (The AI's response to evaluate):\n```\n{candidate_answer}\n```\n\n"
    "**Output Format:**\n"
    "Please respond **ONLY** with a JSON object containing two keys:\n"
    "1. `\"is_judged_correct\"`: A boolean value (true if the Candidate Answer is largely consistent with the core correct information and factually sound, false otherwise).\n"
    "2. `\"reasoning\"`: A brief explanation, highlighting key consistencies or discrepancies regarding factual content.\n\n"
    "Now, provide your evaluation based on these L3 criteria (largely consistent and factually sound result):"
)


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