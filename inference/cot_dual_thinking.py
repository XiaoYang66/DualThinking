"""
Dual Thinking Chain-of-Thought (CoT) System Implementation

This module implements a three-stage dual thinking system for mathematical problem solving:
1. Thinking Outline: Generate step-by-step solution outline
2. Thinking Scores: Evaluate difficulty of each step
3. Thinking Details: Generate detailed reasoning based on difficulty scores

The system adaptively chooses between thinking and non-thinking modes based on step difficulty.
"""

import json
import os
import re
import random

from inference.qwen import request_qwen
from inference.util import extract_final_answer


# Constants
OUTLINE_MAX_TOKENS = 32768
SCORE_MAX_TOKENS = 64
TOP_K_DEFAULT = 20
MAX_RETRY_ATTEMPTS = 3

# 指令模板常量
OUTLINE_INSTRUCTION_v1 = """You are a helpful assistant that generates step-by-step problem-solving steps for mathematical problems. Provide only the logical sequence of steps needed to solve the problem.

Requirements:
1. Focus on WHAT to achieve in each step, not HOW to do it - avoid specifying particular theorems, formulas, or methods unless absolutely necessary
2. Steps must follow logical order - each step should build upon previous results or be independently executable
3. Each step must have a clear, specific target/goal that contributes to the final answer
4. Do not solve the problem or provide numerical results - only the steps needed to reach the solution
5. If new variables or points are introduced for problem-solving convenience, mention them clearly
6. Ensure the sequence of steps leads logically to the final answer requested in the problem
7. Keep steps concise and action-oriented, focusing on objectives rather than methods.

Format: Provide steps in JSON format as {\"step1\": \"description\", \"step2\": \"description\", ...}

Provide only the JSON output with no additional explanation or content."""

OUTLINE_INSTRUCTION = """You are a helpful and brilliant math teacher that decompose math problems into a sequence of clear, manageable subproblems. Analyze the problem and decompose it into a sequence of clear, manageable subproblems that build logically toward the final solution.

Requirements:
1. Break down the main problem into sequential, achievable subgoals where each subproblem is simpler than the original
2. Focus on WHAT to achieve in each step, not HOW to do it - avoid specifying particular theorems, formulas, or methods unless absolutely necessary
3. Each step should either build upon previous results or be independently executable in logical order
4. Every step must have a specific, clear target/goal that directly contributes to reaching the final answer
5. Do not solve the problem or provide numerical results - only identify the key steps needed
6. If new variables, points, or concepts need to be introduced for problem-solving convenience, mention them explicitly
7. Ensure the complete sequence creates a logical pathway from the given information to the requested solution
8. Keep steps concise and action-oriented, focusing on objectives rather than implementation methods

Decomposition Strategy: Identify what intermediate results or insights are needed to answer the main question, then organize these into a logical sequence where each step enables subsequent progress.

Format: Provide steps in JSON format as {\"step1\": \"description\", \"step2\": \"description\", ...}

Provide only the JSON output with no additional explanation or content."""

OUTLINE_INSTRUCTION_SELECTION = """You are a helpful and brilliant math teacher that can help students to solve mathematical problems step by step. I will provide you with different versions of the problem-solving steps, and you need to select the best version that can solve the problem correctly. Please evaluate the quality of each version and return the one that is most likely to lead to the correct solution.
Requirements:
1. Evaluate the quality of each version based on the following criteria:
   - Logical correctness: Does the version follow a logical sequence that leads to the final answer?
   - Clarity: Is the version clear and easy to understand?
   - Completeness: Does the version cover all necessary steps to solve the problem?
   - Relevance: Does the version focus on the key aspects of the problem without unnecessary details?
   - Simplicity(I): Is the version simple and straightforward, avoiding unnecessary complexity?
   - Simplicity(II): Focus on WHAT to achieve in each step, not HOW to do it - avoid specifying particular theorems, formulas, or methods. Only describe the target of each step.
2. Select the version that best meets these criteria and is most likely to lead to the correct solution.
3. Only return the selected version ID. such as version1, version2, etc.
4. No additional comments or explanations. Only return the selected version ID.
"""

SCORE_INSTRUCTION = """You are a helpful and brilliant math teacher that can solve mathematical problems step by step. I will provide you with a problem, the current solution process, and the next step to be solved. Please evaluate the difficulty of this next step and return a value between 0 and 1. A score of 0-0.5 indicates the step is relatively simple—elementary or middle school students with relevant knowledge can solve it quickly without assistance. A score of 0.5-1 indicates the step is difficult—even high school or college students with the necessary knowledge would need extensive thinking or help from others, or may not be able to solve it at all. Requirements: 1.only return a score between 0 and 1. 2. Do not provide any other content or explanation."""

DETAIL_INSTRUCTION = (
    """Please reason step by step, and put your final answer within \\boxed{}."""
)

# Format constants
JSON_CODE_BLOCK_START = "```json"
JSON_CODE_BLOCK_END = "```"
STEP_KEY_PREFIX = "step"


def fix_single_backslashes_regex(text: str) -> str:
    """Fix single backslashes to double backslashes for JSON parsing."""
    return re.sub(r"(?<!\\)\\(?!\\)", r"\\\\", text)


class System3Base:
    """
    System3Base: 双思维链式推理系统基类

    这是一个三阶段的数学问题求解系统：
    1. thinking_outline: 生成解题步骤大纲
    2. thinking_scores: 评估每个步骤的难度
    3. thinking_details: 根据难度生成详细推理过程

    系统会根据步骤难度自适应地选择是否使用thinking模式，
    为困难步骤提供更深入的推理过程。

    子类应覆盖 get_thinking_request_func() 和 get_non_thinking_request_func()
    以使用不同的模型。

    Attributes:
        question (dict): 包含问题信息的字典
        args: 命令行参数对象
        steps (dict): 解题步骤大纲
        steps_scores (dict): 每个步骤的难度评分
        steps_detail (dict): 每个步骤的详细推理过程
        steps_content (dict): 每个步骤的最终答案内容
        threshold (float): 难度阈值，决定是否使用thinking模式
    """

    def __init__(self, question: dict, args):
        """Initialize System3Base instance."""
        self.question = question
        self.args = args
        self.steps: dict = {}
        self.steps_scores: dict = {}
        self.steps_detail: dict = {}
        self.steps_content: dict = {}
        self.threshold: float = args.difficulty
        self.step_outline_versions: int = 8

    def get_thinking_request_func(self):
        """Get thinking mode request function. Subclasses can override."""
        return request_qwen

    def get_non_thinking_request_func(self):
        """Get non-thinking mode request function. Subclasses can override."""
        return request_qwen

    def thinking_outline(self) -> None:
        """Phase 1: Generate step-by-step solution outline."""
        steps_exist, content = self.check_field_exist("steps")
        if steps_exist and content is not None:
            print("steps already exist, skip thinking outline.")
            self.steps = content
            return

        content = self.thinking_outline_single(generate_id=self.question["sampling_id"])
        self.steps = json.loads(content)
        self.steps[f"step{len(self.steps)}"] = self.question["question"]

    def thinking_outline_think_mode(self) -> None:
        """
        阶段1：生成数学问题的逐步解题大纲

        此方法使用thinking模式生成问题的解题步骤大纲。
        生成的步骤必须符合JSON格式：{"step1": "描述1", "step2": "描述2", ...}

        生成的大纲要求：
        - 步骤逻辑顺序正确
        - 每个步骤目标明确
        - 步骤间相互依赖或可并行执行
        - 不提供具体解答，只提供方法论

        Raises:
            json.JSONDecodeError: 如果生成的内容无法解析为JSON

        Note:
            此方法会重试直到生成有效的JSON格式内容
        """

        instruction = OUTLINE_INSTRUCTION
        question_content = self.question["problem"]

        messages = [
            {
                "role": "system",
                "content": instruction,
            },
            {"role": "user", "content": question_content},
        ]

        extra_body = {}

        def check_json_format(content: str) -> bool:
            """
            检查内容是否为有效的JSON格式

            验证生成的内容是否：
            1. 可以被JSON解析
            2. 包含正确格式的步骤键（stepN，N为数字）

            Args:
                content (str): 需要检查的内容字符串

            Returns:
                bool: 如果格式有效返回True，否则返回False
            """
            # Fix single backslashes in the content
            content = fix_single_backslashes_regex(content)

            try:
                data_ = json.loads(content)
                for key in data_.keys():
                    # check if the key align with the format of 'stepN',where N is a number
                    if not key.startswith(STEP_KEY_PREFIX) or not key[4:].isdigit():
                        print(
                            f"Invalid key format: {key}. Expected format is '{STEP_KEY_PREFIX}N' where N is a number."
                        )
                        return False
                return True
            except json.JSONDecodeError:
                return False

        content = None
        # 重试循环：直到生成有效的JSON格式内容
        while True:
            request_func = self.get_thinking_request_func()
            results = request_func(
                extra_body=extra_body,
                thinking=True,  # 使用thinking模式，生成大纲需要深度推理
                messages=messages,
                max_tokens=OUTLINE_MAX_TOKENS,  # 使用常量定义的大纲生成token数
                top_k=TOP_K_DEFAULT,  # 使用常量定义的top_k值
                sampling_seed=self.question[
                    "sampling_id"
                ],  # 使用问题的sampling_id作为随机种子
            )

            # 提取生成的内容
            content = results["content"]
            print("thinking outline content:", results["reasoning_content"])
            print("---" * 20)
            print(content)

            # 尝试从markdown代码块中提取JSON内容
            content = (
                content.split(JSON_CODE_BLOCK_START)[-1]
                .split(JSON_CODE_BLOCK_END)[0]
                .strip()
            )

            if check_json_format(content):
                break
            print("outline stage: The generated content is not in valid JSON format, retrying...")

        # 修复反斜杠并解析JSON
        content = fix_single_backslashes_regex(content)
        self.steps = json.loads(content)

    def thinking_outline_single(self, generate_id) -> str:
        """
        generate single versions of outline
        """
        instruction = OUTLINE_INSTRUCTION
        question_content = self.question["problem"]

        messages = [
            {
                "role": "system",
                "content": instruction,
            },
            {"role": "user", "content": question_content},
        ]

        extra_body = {}

        def check_json_format(content: str) -> bool:
            """
            检查内容是否为有效的JSON格式

            验证生成的内容是否：
            1. 可以被JSON解析
            2. 包含正确格式的步骤键（stepN，N为数字）

            Args:
                content (str): 需要检查的内容字符串

            Returns:
                bool: 如果格式有效返回True，否则返回False
            """
            # Fix single backslashes in the content
            content = fix_single_backslashes_regex(content)

            try:
                data_ = json.loads(content)
                for key in data_.keys():
                    # check if the key align with the format of 'stepN',where N is a number
                    if not key.startswith(STEP_KEY_PREFIX) or not key[4:].isdigit():
                        print(
                            f"Invalid key format: {key}. Expected format is '{STEP_KEY_PREFIX}N' where N is a number."
                        )
                        return False
                return True
            except json.JSONDecodeError:
                print(f"Invalid JSON format: {content}, parse json decode error.")
                return False

        content = None
        # 重试循环：直到生成有效的JSON格式内容
        while True:
            request_func = self.get_non_thinking_request_func()
            results = request_func(
                extra_body=extra_body,
                thinking=False,  # 使用非thinking模式，生成大纲不需要深度推理
                messages=messages,
                max_tokens=OUTLINE_MAX_TOKENS,  # 使用常量定义的大纲生成token数
                top_k=TOP_K_DEFAULT,  # 使用常量定义的top_k值
                sampling_seed=generate_id
                * random.randint(1, 1000),  # 使用问题的sampling_id作为随机种子
            )

            # 提取生成的内容
            content = results["content"]
            # 尝试从markdown代码块中提取JSON内容
            content = (
                content.split(JSON_CODE_BLOCK_START)[-1]
                .split(JSON_CODE_BLOCK_END)[0]
                .strip()
            )

            # 检查内容是否为有效JSON格式
            if check_json_format(content):
                break
            
            print(content)
            print("outline stage: The generated content is not in valid JSON format, retrying...")

        content = fix_single_backslashes_regex(content)
        return content

    def _check_version_format(self, content: str) -> bool:
        """Validate version format (e.g., 'version1', 'version2')."""
        if not content.startswith("version"):
            print(f"Invalid version format: {content}. Expected format is 'versionN' where N is a number.")
            return False

        version_number = content[7:]
        if not version_number.isdigit():
            print(f"Invalid version number: {version_number}. Expected a number.")
            return False

        version_num = int(version_number)
        if not 1 <= version_num <= self.step_outline_versions:
            print(f"Invalid version number: {version_number}. Expected between 1 and {self.step_outline_versions}.")
            return False

        return True

    def thinking_outline_majority_voting(self) -> None:
        """Generate multiple outline versions and select the best one."""
        steps_exist, content = self.check_field_exist("steps")
        if steps_exist and content is not None:
            print("steps already exist, skip thinking outline.")
            self.steps = content
            return

        # Generate multiple outline versions
        outlines = {}
        for version_id in range(1, self.step_outline_versions + 1):
            version_temp = self.thinking_outline_single(
                generate_id=version_id * self.question["sampling_id"]
            )
            outlines[f"version{version_id}"] = json.loads(version_temp)

        # Format versions for evaluation
        out_versions_content = "\n".join([
            f"version{i}: {json.dumps(outlines[f'version{i}'])}"
            for i in range(1, self.step_outline_versions + 1)
        ])

        messages = [
            {"role": "system", "content": OUTLINE_INSTRUCTION_SELECTION},
            {
                "role": "user",
                "content": f"Evaluate the following outline versions and select the best one:\nQuestion: {self.question['problem']}\n\n{out_versions_content}",
            },
        ]

        # Try to get valid version selection
        for _ in range(MAX_RETRY_ATTEMPTS):
            request_func = self.get_non_thinking_request_func()
            results = request_func(
                extra_body={},
                thinking=False,
                messages=messages,
                max_tokens=OUTLINE_MAX_TOKENS,
                top_k=TOP_K_DEFAULT,
                sampling_seed=self.question["sampling_id"] * random.randint(1, 1000),
            )

            content = results["content"].strip()

            if self._check_version_format(content):
                self.steps = outlines[content]
                print(f"Selected outline version: {self.steps}")
                self.steps[f"step{len(self.steps)}"] = self.question["question"]
                return
            
            print(content)
            print("Invalid version format, retrying...")

        print(f"Failed to generate valid version after {MAX_RETRY_ATTEMPTS} attempts.")
        self.steps = {}
        return None

    def thinking_scores(self) -> None:
        """Phase 2: Evaluate difficulty of each step (0-1 scale)."""
        step_scores_exist, content = self.check_field_exist("steps_scores")
        if step_scores_exist and content is not None:
            print("steps_scores already exist, skip thinking scores.")
            self.steps_scores = content
            return

        for next_step_key in range(1, len(self.steps) + 1):
            step_key = f"{STEP_KEY_PREFIX}{next_step_key}"
            next_step = self.steps[step_key]
            current_process = " ".join([
                self.steps[f"{STEP_KEY_PREFIX}{i}"]
                for i in range(1, int(next_step_key))
            ])

            query = f"problem: {self.question['problem']}\ncurrent solution process: {current_process}\nnext step to be solved:{next_step}"
            messages = [
                {"role": "system", "content": SCORE_INSTRUCTION},
                {"role": "user", "content": query}
            ]

            while True:
                request_func = self.get_non_thinking_request_func()
                results = request_func(
                    extra_body={},
                    thinking=False,
                    messages=messages,
                    max_tokens=SCORE_MAX_TOKENS,
                    top_k=TOP_K_DEFAULT,
                    sampling_seed=self.question["sampling_id"],
                )

                content = results["content"]
                try:
                    score = float(content.strip())
                    if 0 <= score <= 1:
                        self.steps_scores[step_key] = score
                        break
                    print(f"Score {score} out of range [0,1], retrying...")
                except ValueError:
                    print(f"Error converting content to float: {content.strip()}, retrying")
        
        # Ensure final step uses thinking mode
        last_step_key = f"{STEP_KEY_PREFIX}{len(self.steps)}"
        self.steps_scores[last_step_key] = 0.9

    def thinking_details(self) -> None:
        """Phase 3: Generate detailed reasoning based on difficulty scores."""
        conditions = self.question["condition"]

        for next_step_key in range(1, len(self.steps) + 1):
            step_key = f"{STEP_KEY_PREFIX}{next_step_key}"
            next_step = self.steps[step_key]

            current_process = " ".join([
                f"{self.steps[f'{STEP_KEY_PREFIX}{i}']}\nstep answer:{self.steps_content[f'{STEP_KEY_PREFIX}{i}']}"
                for i in range(1, int(next_step_key))
            ])

            query = f"Given conditions: {conditions}\n Current process: {current_process}\n Question: {next_step}"

            enable_thinking = self.steps_scores[step_key] > self.threshold
            mode = "thinking" if enable_thinking else "normal"
            print(f"Step {next_step_key}: Using {mode} mode (difficulty: {self.steps_scores[step_key]:.2f})")

            messages = [
                {"role": "system", "content": DETAIL_INSTRUCTION},
                {"role": "user", "content": query}
            ]

            request_func = self.get_thinking_request_func() if enable_thinking else self.get_non_thinking_request_func()
            
            results = request_func(
                extra_body={},
                thinking=enable_thinking,
                messages=messages,
                max_tokens=self.args.max_tokens,
                top_k=self.args.top_k,
                sampling_seed=self.question["sampling_id"],
            )

            self.steps_detail[step_key] = results["reasoning_content"]
            self.steps_content[step_key] = results["content"]

    def save_process(self) -> None:
        """Save the complete reasoning process to JSON file."""
        output_path = os.path.join(
            self.args.output_folder,
            self.args.cot_name,
            self.args.prompt_version,
            f"{self.args.max_tokens}",
            f"{self.args.difficulty}",
            f"{self.question['id']}_{self.question['sampling_id']}.json",
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        self.question.update({
            "cot_name": self.args.cot_name,
            "model_name": self.args.model_name,
            "steps": self.steps,
            "steps_scores": self.steps_scores,
            "steps_detail": self.steps_detail,
            "steps_content": self.steps_content,
        })

        with open(output_path, "w", encoding="UTF-8") as f:
            json.dump(self.question, f, indent=4, ensure_ascii=False)

        print(f"Inference completed for question {self.question['id']}_{self.question['sampling_id']}. "
              f"Output saved to {output_path}.")

    def _get_output_path(self) -> str:
        """Get the output file path for current question."""
        return os.path.join(
            self.args.output_folder,
            self.args.cot_name,
            self.args.prompt_version,
            f"{self.args.max_tokens}",
            f"{self.args.difficulty}",
            f"{self.question['id']}_{self.question['sampling_id']}.json",
        )

    def check_exist(self) -> bool:
        """Check if output file already exists with valid results."""
        output_path = self._get_output_path()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if not os.path.exists(output_path):
            print(f"Starting inference for question {self.question['id']}_{self.question['sampling_id']}_"
                  f"{self.args.max_tokens}_{self.args.cot_name}")
            return False

        try:
            with open(output_path, "r", encoding="UTF-8") as f:
                existing_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {output_path}")
            return False

        if existing_data.get("final_answer") is not None and existing_data.get("is_correct") is not None:
            print(f"Skipping inference for question {self.question['id']}_{self.question['sampling_id']}_"
                  f"{self.args.max_tokens}_{self.args.cot_name} (file already exists).")
            return True
        
        print(f"File {output_path} exists but does not contain final answer or correctness, reprocessing...")
        return False

    def check_field_exist(self, field: str) -> tuple:
        """Check if specified field exists in output file.
        
        Returns:
            tuple: (exists: bool, field_value: any)
        """
        output_path = self._get_output_path()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if not os.path.exists(output_path):
            return False, None

        try:
            with open(output_path, "r", encoding="UTF-8") as f:
                existing_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {output_path}")
            return False, None

        if existing_data.get(field) is not None:
            print(f"Field {field} already exists in {output_path}, skipping inference.")
            return True, existing_data[field]
        
        print(f"Field {field} does not exist in {output_path}")
        return False, None

    def extract_final_answer(self) -> None:
        """Extract final answer from the last step's response."""
        try:
            last_step_key = f"{STEP_KEY_PREFIX}{len(self.steps)}"
            response = self.steps_content[last_step_key]
            final_answer, is_correct = extract_final_answer(response, self.question["answer"])
            self.question["final_answer"] = final_answer
            self.question["is_correct"] = is_correct
            print(f"Final answer extracted: {final_answer}")
        except (ValueError, KeyError) as e:
            print(f"Error extracting final answer for question {self.question['id']}_"
                  f"{self.question['sampling_id']}: {e}")
            self.question["final_answer"] = None

    def get_step_summary(self) -> dict:
        """Get summary of steps including difficulty distribution."""
        if not self.steps:
            return {"total_steps": 0}

        total_steps = len(self.steps)
        if not self.steps_scores:
            return {
                "total_steps": total_steps,
                "average_difficulty": 0,
                "hard_steps": 0,
                "easy_steps": 0,
                "thinking_threshold": self.threshold,
            }

        difficulties = list(self.steps_scores.values())
        avg_difficulty = sum(difficulties) / len(difficulties)
        hard_steps = sum(1 for score in difficulties if score > self.threshold)
        
        return {
            "total_steps": total_steps,
            "average_difficulty": avg_difficulty,
            "hard_steps": hard_steps,
            "easy_steps": total_steps - hard_steps,
            "thinking_threshold": self.threshold,
        }

    def set_difficulty_threshold(self, threshold: float) -> None:
        """Set difficulty threshold for thinking mode."""
        if not 0 <= threshold <= 1:
            raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
        self.threshold = threshold
        print(f"Difficulty threshold updated to {threshold}")

    def validate_question_format(self) -> bool:
        """Validate question has required fields."""
        required_fields = ["id", "sampling_id", "problem"]
        for field in required_fields:
            if field not in self.question:
                print(f"Missing required field: {field}")
                return False
        return True

    def is_solved(self) -> bool:
        """Check if question has been solved with valid answer."""
        return (
            self.question.get("final_answer") is not None
            and self.question.get("is_correct") is not None
        )


def inference_sysytem3(question: dict, args) -> bool:
    """Main entry point for System3 dual thinking inference.

    Args:
        question: Dictionary containing problem info (id, sampling_id, problem, condition)
        args: Configuration object with output_folder, max_tokens, cot_name, model_name, top_k

    Returns:
        bool: True if inference completed successfully, False otherwise
    """
    system3 = System3Base(question, args)

    if system3.check_exist():
        return True

    print("Starting System3 dual thinking inference...")

    print("Phase 1: Generating step outline...")
    system3.thinking_outline_majority_voting()

    if len(system3.steps) == 0:
        print(f"Question {question['id']}_{question['sampling_id']} could not be solved. No steps generated.")
        return False

    print("Phase 2: Evaluating step difficulties...")
    system3.thinking_scores()

    print("Phase 3: Generating detailed reasoning...")
    system3.thinking_details()

    print("Extracting final answer...")
    system3.extract_final_answer()

    print("Saving results...")
    system3.save_process()

    if not system3.is_solved():
        print(f"Question {question['id']}_{question['sampling_id']} could not be solved. No final answer extracted.")
        return False
    
    print(f"Question {question['id']}_{question['sampling_id']} solved successfully! "
          f"Final answer: {system3.question['final_answer']}")
    return True


if __name__ == "__main__":
    pass
