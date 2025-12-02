"""
Dual Thinking Chain-of-Thought (CoT) System - QwQ Variant

This module extends System3Base to use QwQ model for thinking mode
and Qwen for non-thinking mode.
"""

import json
import os

from inference.qwen import request_qwen
from inference.qwq import request_qwq
from inference.util import extract_final_answer
from inference.cot_dual_thinking import (
    System3Base,
    DETAIL_INSTRUCTION,
    STEP_KEY_PREFIX,
)

SUMMARIZE = """Please list what you have achieved in your last response. Note that you should only output the summarization. You should list all the key steps and important intermediate conclusion. Please list them with '*.' """


class System3QwQ(System3Base):
    """
    System3QwQ: 基于 System3Base 的 QwQ 模型变体

    使用 QwQ 模型进行 thinking 模式推理，
    使用 Qwen 进行 non-thinking 模式推理。
    
    额外特性：
    - steps_summary: 由于上下文窗口较小，需要总结步骤内容
    """

    def __init__(self, question: dict, args):
        """
        初始化System3QwQ实例

        Args:
            question (dict): 问题字典，包含id、problem、conditions等字段
            args: 包含配置参数的对象，如max_tokens、top_k、output_folder等
        """
        super().__init__(question, args)
        # 由于上下文窗口大小较小，需要总结步骤内容
        self.steps_summary = {}  # 步骤总结：{"step1": "总结1", "step2": "总结2", ...}

    def get_thinking_request_func(self):
        """返回 QwQ 模型的 thinking 函数"""
        return request_qwq

    def get_non_thinking_request_func(self):
        """返回 Qwen 模型的 non-thinking 函数"""
        return request_qwen

    def thinking_details(self) -> None:
        """
        阶段3：根据难度评分生成每个步骤的详细推理过程（QwQ变体）

        覆盖基类方法，使用步骤总结替代完整内容以减少上下文长度。
        """
        instruction = DETAIL_INSTRUCTION
        conditions = self.question["condition"]

        # 依次处理每个步骤
        for next_step_key in range(1, len(self.steps) + 1):
            step_key = f"{STEP_KEY_PREFIX}{next_step_key}"
            next_step = self.steps[step_key]

            # 构建当前进度：使用总结替代完整内容以减少token
            current_process = ""
            for i in range(1, int(next_step_key)):
                current_process += (
                    self.steps[f"{STEP_KEY_PREFIX}{i}"]
                    + "\nstep answer:"
                    + self.steps_summary[f"{STEP_KEY_PREFIX}{i}"]
                    + "\n"
                )

            # 构建查询：给定条件 + 当前进度 + 要解决的问题
            query = f"{instruction}\nGiven conditions: {conditions}\n Current process: {current_process}\n Question: {next_step}"

            # 根据难度分数决定是否使用thinking模式
            if self.steps_scores[step_key] > self.threshold:
                enable_thinking = True
                print(
                    f"Step {next_step_key}: Using thinking mode (difficulty: {self.steps_scores[step_key]:.2f})"
                )
            else:
                enable_thinking = False
                print(
                    f"Step {next_step_key}: Using normal mode (difficulty: {self.steps_scores[step_key]:.2f})"
                )

            messages = [
                {"role": "user", "content": query},
            ]

            extra_body = {}

            # 根据难度选择request函数
            if enable_thinking:
                request_func = self.get_thinking_request_func()
            else:
                request_func = self.get_non_thinking_request_func()

            results = request_func(
                extra_body=extra_body,
                thinking=enable_thinking,
                messages=messages,
                max_tokens=self.args.max_tokens,
                top_k=self.args.top_k,
                sampling_seed=self.question["sampling_id"],
            )

            # 保存推理过程和最终答案
            self.steps_detail[step_key] = results["reasoning_content"]
            self.steps_content[step_key] = results["content"]

            # 总结步骤内容
            step_answer, _ = extract_final_answer(
                self.steps_content[step_key],
                None,
            )
            self.steps_summary[step_key] = step_answer

    def save_process(self) -> None:
        """
        保存完整的推理过程到JSON文件（QwQ变体）

        覆盖基类方法以包含 steps_summary 字段。
        """
        # 构建输出文件路径
        output_path = os.path.join(
            self.args.output_folder,
            self.args.cot_name,
            self.args.prompt_version,
            f"{self.args.max_tokens}",
            f"{self.args.difficulty}",
            f"{self.question['id']}_{self.question['sampling_id']}.json",
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 添加系统相关信息到问题字典
        self.question["cot_name"] = self.args.cot_name
        self.question["model_name"] = self.args.model_name
        self.question["steps"] = self.steps
        self.question["steps_scores"] = self.steps_scores
        self.question["steps_detail"] = self.steps_detail
        self.question["steps_content"] = self.steps_content
        self.question["steps_summary"] = self.steps_summary  # 额外字段

        # 保存到JSON文件
        with open(output_path, "w", encoding="UTF-8") as f:
            json.dump(self.question, f, indent=4, ensure_ascii=False)

        print(
            f"Inference completed for question {self.question['id']}_{self.question['sampling_id']}. "
            f"Output saved to {output_path}."
        )


def inference_sysytem3(question: dict, args) -> bool:
    """
    System3QwQ双思维推理系统的主入口函数
    """
    system3 = System3QwQ(question, args)

    if system3.check_exist():
        return True

    print("Starting System3QwQ dual thinking inference...")

    print("Phase 1: Generating step outline...")
    system3.thinking_outline_majority_voting()

    if len(system3.steps) == 0:
        print(
            f"Question {question['id']}_{question['sampling_id']} could not be solved. No steps generated."
        )
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
        print(
            f"Question {question['id']}_{question['sampling_id']} could not be solved. No final answer extracted."
        )
        return False
    else:
        print(
            f"Question {question['id']}_{question['sampling_id']} solved successfully! Final answer: {system3.question['final_answer']}"
        )
        return True


if __name__ == "__main__":
    pass
