import json
import os

# from inference.qwen import request_qwen
from inference.qwen2_5 import request_qwen2  # think mode using deepseek
from inference.util import extract_final_answer
from inference.cot_instruction import (
    aime24_instruction,
    aime25_instruction,
    amc23_instruction,
)


# cot_name explain
# system1: normal chain of thought without thinking
# system2: chain of thought with thinking
# system3: dual thinking chain of thought, only key steps using thinking mode
# nothinking: Reasoning Models Can Be Effective Without Thinking


def inference_sysytem1_2(question: dict, enable_thinking, args):
    """
    Function to run inference on a single question.
    """
    max_tokens = args.max_tokens
    stop_words = args.stop_words
    temperature = args.temperature
    top_p = args.top_p
    top_k = args.top_k
    dataset_name = args.dataset_name

    # Create the output folder if it does not exist
    output_folder = args.output_folder
    # output_folder = os.path.join(output_folder, args.model_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    # Check if the output file already exists
    # If it exists, skip the inference for this question
    output_path = os.path.join(
        output_folder,args.cot_name,args.prompt_version,str(args.max_tokens),str(args.difficulty),
        f"{question['id']}_{question['sampling_id']}.json",
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="UTF-8") as output_file:
            existing_question = json.load(output_file)
        # Check if the existing question has response, final_answer, and is_correct
        # If all these fields are present, we can skip the inference
        # This is to avoid re-running inference for already processed questions 
        if existing_question.get("response") is not None and existing_question.get("final_answer") is not None and existing_question.get("is_correct") is not None:
            
            print(
                f"Output file {output_path} already exists. Skipping inference for question {question['id']}_{question['sampling_id']}_{args.max_tokens}_{args.cot_name}."
            )
            return True

    if "Qwen2.5" in args.model_name:
        # If the model is Qwen2.5, use the request_qwen2 function
        request_func = request_qwen2
    elif "llama" in args.model_name:
        # If the model is Llama, you can define another request function here
        # For now, we will just use the same function as a placeholder
        request_func = request_qwen2
    else:
        # If the model is not Qwen2.5, you can define another request function here
        # For now, we will just use the same function as a placeholder
        raise NotImplementedError("Only Qwen2.5 model is supported in this script.")

    question_content = question["problem"] 
    

    if dataset_name == "AIME25":
        # For AIME25, use the specific instruction
        data_instruction = aime25_instruction
    elif dataset_name == "AIME24":
        # For AIME24, use the specific instruction
        data_instruction = aime24_instruction
    elif dataset_name == "AMC23":
        # For AMC23, use the specific instruction
        data_instruction = amc23_instruction
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")

    messages = [
        # {
        #     "role": "system",
        #     "content": data_instruction, # not recommend to use system message here, because it will cause the model to generate invalid json format
        # },
        {"role": "user", "content": f"{data_instruction}\n{question_content}"},
    ]

    extra_body = {}

    response = request_func(
        extra_body=extra_body,
        thinking=enable_thinking,
        max_tokens=max_tokens,
        messages=messages,
        stop_words=stop_words,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        sampling_seed=question["sampling_id"],
    )

    question["response"] = response
    question["cot_name"] = args.cot_name
    question["model_name"] = args.model_name

    # extract the final answer from the response
    try:
        true_answer = question.get("answer")
        if true_answer is not None:
            extracted_answer, is_correct = extract_final_answer(response, true_answer)
            question["final_answer"] = extracted_answer
            question["is_correct"] = is_correct
        else:
            # If no true answer available, just extract without validation
            question["final_answer"] = None
            question["is_correct"] = None
    except ValueError as e:
        print(
            f"Error extracting final answer for question {question['id']}_{question['sampling_id']}: {e}"
        )
        question["final_answer"] = None
        question["is_correct"] = None

    # Save the question with response to the output file
    with open(output_path, "w", encoding="UTF-8") as output_file:
        json.dump(question, output_file, indent=4, ensure_ascii=False)

    print(
        f"Inference completed for question {question['id']}_{question['sampling_id']}. Output saved to {output_path}."
    )
    if question["is_correct"] is not None and question["is_correct"] and question["final_answer"] is not None:
        return True
    else:
        return False


# def inference_sysytem2(question: dict, args):
#     """
#     Function to run inference on a single question.
#     """
#     return inference_sysytem1_2(question, enable_thinking=True, args=args)


def inference_sysytem1(question: dict, args):
    """
    Function to run inference on a single question.
    """
    return inference_sysytem1_2(question, enable_thinking=False, args=args)
