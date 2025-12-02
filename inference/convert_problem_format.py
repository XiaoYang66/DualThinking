import json
import re
import argparse
from inference.qwen import request_qwen

def convert_problem_format(input_file_path: str) -> list:
    """
    Convert problems from original format to conditions + question format.
    
    Args:
        input_file_path: Path to the input JSONL file containing problems
        
    Returns:
        List of converted problems
    """
    instruction = ("I will provide you with some problems. Please convert these problems into a clear format of "
                  "conditions + question. The conditions should be the given information, and the question should "
                  "be the question to be answered. The output format should be a json object with two keys: "
                  "'conditions' and 'question': {'conditions': '', 'question': ''}")
    
    # Read problems from jsonl file
    problems = []
    with open(input_file_path, 'r', encoding='UTF-8') as file:
        for line in file:
            problem = line.strip()
            if problem:  # Skip empty lines
                print(problem)
                problem = json.loads(problem)
                problems.append(problem)
    
    for problem in problems:
        if "conditions" in problem and "question" in problem:
            # If the problem already has conditions and question, skip it
            continue
            
        messages = [
            {
                "role": "system",
                "content": instruction,
            },
            {
                "role": "user",
                "content": problem["problem"] if "problem" in problem else problem["question"],
            },
        ]
        
        extract_body = {}
        response = request_qwen(
            extra_body=extract_body,
            thinking=False,  # 使用非thinking模式
            messages=messages,
            max_tokens=1024,
            top_k=20,
        )
        
        content = response["content"]
        
        # Extract JSON from response
        if "```json" in content:
            try:
                content = content.split("```json")[1].split("```")[0].strip()
            except IndexError:
                print(f"Error extracting JSON from response for problem {problem.get('id', 'unknown')}")
                continue
        
        try:
            # Try to parse the content as json
            cont_json = json.loads(content)
        except json.JSONDecodeError as e:
            # Try to fix LaTeX escape issues in JSON
            fixed_content = content  # Initialize fixed_content
            try:
                # Fix LaTeX backslashes by properly escaping them for JSON
                # This regex finds LaTeX commands like \frac, \sqrt, \angle, etc.
                # and ensures they are properly escaped for JSON
                fixed_content = re.sub(r'\\([a-zA-Z]+)', r'\\\\\\1', content)
                # Also fix standalone backslashes before non-alphabetic characters
                fixed_content = re.sub(r'\\(?![a-zA-Z\\"])', r'\\\\', fixed_content)
                cont_json = json.loads(fixed_content)
                print(f"Fixed LaTeX escaping for problem {problem.get('id', 'unknown')}")
            except json.JSONDecodeError:
                print(f"Error decoding JSON even after escape fix: {e}")
                print(f"Problem ID: {problem.get('id', 'unknown')}")
                print("-" * 20)
                print(f"Original Content: {content}")
                print(f"Fixed Content: {fixed_content}")
                print("-" * 20)
                continue
        
        if isinstance(cont_json, dict):
            problem["conditions"] = cont_json.get("conditions", "")
            problem["question"] = cont_json.get("question", "")
        else:
            print(f"Warning: Response is not a dictionary for problem {problem.get('id', 'unknown')}")
    
    # Save the problems to the same file (overwrite)
    with open(input_file_path, 'w', encoding='UTF-8') as file:
        for problem in problems:
            file.write(json.dumps(problem, ensure_ascii=False) + '\n')
    
    return problems
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert problems to conditions + question format")
    parser.add_argument("input", help="Path to input JSONL file")
    args = parser.parse_args()
    convert_problem_format(args.input)