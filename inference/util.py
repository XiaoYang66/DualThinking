"""
Utility helpers used by inference scripts.

- Extract/validate final answers from LLM responses
- Simple service readiness checks
- Lightweight folder copy + JSON field pruning

CLI usage (examples):

  - Check port availability:
      python -m inference.util check-port --host localhost --port 50073

  - Wait until service is ready:
      python -m inference.util wait-service --host localhost --port 50073 --interval 5

  - Copy a folder and prune heavy fields from JSON files:
      python -m inference.util copy-folder --src /path/src --dst /path/dst \
        --prune steps_detail steps_content final_answer is_correct steps_summary
"""

import argparse
import logging
import socket
import time
from typing import Dict, Tuple, Union, List
import os
import json
import shutil

from inference.qwen import request_qwen

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_final_answer(
    response: Union[Dict, str], true_answer: Union[str, int, float,None]
) -> Tuple[str, Union[bool, None]]:
    """
    Extract final answer from response and validate against true answer.

    Args:
        response: LLM response (dict with 'content' key or string)
        true_answer: The correct answer to compare against

    Returns:
        Tuple of (extracted_answer, is_correct)
    """
    if isinstance(response, dict):
        content = response["content"]
    else:
        content = response

    return extract_by_llm(content, true_answer=true_answer)


def _make_llm_request(messages: list, max_tokens: int = 64) -> str:
    """
    Helper function to make LLM requests with consistent parameters.

    Args:
        messages: List of message dictionaries
        max_tokens: Maximum tokens for response

    Returns:
        Response content as string
    """
    try:
        response = request_qwen(
            messages=messages, thinking=False, extra_body={}, max_tokens=max_tokens
        )
        return response["content"].strip()
    except Exception as e:
        logger.error("Error making LLM request: %s", e)
        raise


def extract_by_llm(
    content: str, true_answer: Union[str, int, float, None]
) -> Tuple[str, Union[bool, None]]:
    """
    Extract the final answer from content using LLM and validate against true answer.

    Args:
        content: The solution content to extract answer from
        true_answer: The correct answer to validate against

    Returns:
        Tuple of (extracted_answer, is_correct)
    """
    logger.info("Extracting final answer from content using LLM")

    # Extract final answer
    extraction_messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Your task is to extract the final answer "
                "to an integer or float from a solution process. Only return the final "
                "answer as an integer or float. Do not return any other text."
            ),
        },
        {
            "role": "user",
            "content": f"Extract the final answer to an integer or float from the provided solution process: {content}",
        },
    ]

    extracted_answer = _make_llm_request(extraction_messages)
    
    if true_answer is None:
        logger.warning("True answer is None, skipping validation.")
        return extracted_answer, None

    # Validate answer
    validation_messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Your task is to check if the final answer "
                "is correct or not based on the true answer. If the final answer is correct, "
                "return 'correct'. If the final answer is incorrect, return 'incorrect'. "
                "Only return 'correct' or 'incorrect'. Do not return any other text. "
                "Do not consider the difference of float and integer. For example, "
                "if the final answer is 3.0 and the true answer is 3, you should return 'correct'."
            ),
        },
        {
            "role": "user",
            "content": f"Extracted final answer: {extracted_answer}. True answer: {true_answer}.",
        },
    ]

    check_result = _make_llm_request(validation_messages)

    # Convert result to boolean
    if check_result.lower() == "correct":
        is_correct = True
    elif check_result.lower() == "incorrect":
        is_correct = False
    else:
        logger.warning("Unexpected check result: %s", check_result)
        raise ValueError(
            f"Invalid check result: {check_result}. Expected 'correct' or 'incorrect'."
        )

    logger.info("Extracted answer: %s, Correct: %s", extracted_answer, is_correct)
    return extracted_answer, is_correct


def validate_inputs(
    response: Union[Dict, str], true_answer: Union[str, int, float]
) -> None:
    """
    Validate input parameters.

    Args:
        response: LLM response to validate
        true_answer: True answer to validate

    Raises:
        ValueError: If inputs are invalid
    """
    if response is None:
        raise ValueError("Response cannot be None")

    if isinstance(response, dict) and "content" not in response:
        raise ValueError("Response dict must contain 'content' key")

    if isinstance(response, str) and not response.strip():
        raise ValueError("Response string cannot be empty")

    if true_answer is None:
        raise ValueError("True answer cannot be None")


def is_port_available(host: str = "localhost", port: int = 8000) -> bool:
    """
    检查指定端口是否可用（即是否有服务在监听）。

    Args:
        host: 主机地址，默认为 'localhost'
        port: 端口号，默认为 8000

    Returns:
        bool: 如果端口可用（有服务监听），返回 True；否则返回 False
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            result = sock.connect_ex((host, port))
            return result == 0
    except (socket.error, OSError):
        return False


def check_service_ready(host: str = "localhost", port: int = 8000, interval: int = 10) -> bool:
    """
    检查服务是否就绪（立即检查，不等待）。

    Args:
        host: 主机地址，默认为 'localhost'
        port: 端口号，默认为 8000

    Returns:
        bool: 如果服务就绪返回 True，否则返回 False
    """
    while True:
        if is_port_available(host, port):
            logger.info("Service at %s:%s is ready.", host, port)
            return True
        logger.warning("Service at %s:%s is not ready yet. Retrying in %ss ...", host, port, interval)
        time.sleep(max(1, int(interval)))




if __name__ == "__main__":
    pass


