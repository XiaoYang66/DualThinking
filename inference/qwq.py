# import argparse
from gc import enable
from itertools import count
from math import e
from openai import OpenAI
import requests


# Set OpenAI's API key and API base to use vLLM's API server.
OPENAI_API_BASE_THINKING = "http://localhost:50073/v1"
OPENAI_API_BASE_THINKING_TOKENIZED = "http://localhost:50073/tokenize"
OPENAI_API_KEY = "None"  # Not used, but required by the OpenAI client.

DEEPSEEK_MAX_TOKENS = 32768  # Maximum tokens for DeepSeek models


def _make_request_tokenize_count(
    extra_body: dict,
    enable_thinking: bool = False,
    max_tokens: int = 16384,
    messages: list | None = None,
    stop_words: list | None = None,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 40,
    sampling_seed: int = 0,
) -> dict:

    # get model name
    base_url = (
        OPENAI_API_BASE_THINKING # must be thinking mode to get the model name
    )
    client = OpenAI(
        base_url=base_url,
        api_key=OPENAI_API_KEY,
    )
    models = client.models.list()
    model = models.data[0].id

    base_url = (
        OPENAI_API_BASE_THINKING_TOKENIZED # must be thinking mode to get the tokenized count
    )

    # 将messages转换为单个文本字符串
    # text = ""

    if messages is None:
        return {"token_count": 0, "tokens": [], "text": ""}
    # 将messages转换为prompt格式
    prompt = ""
    for msg in messages:
        prompt += f"{msg['role']}: {msg['content']}\n"
    response = requests.post(
        base_url,
        json={"prompt": prompt, "messages": messages, "model": model},
    )

    if response.status_code == 200:
        data = response.json()
        tokens = data.get("tokens", [])
        return {"token_count": len(tokens), "tokens": tokens}
    else:
        print(f"INSIDE DEEPSEEK COUNT TOKENS, Error: {response.status_code} - {response.text}")
        return {"token_count": 0, "tokens": []}


def _make_request_response(
    extra_body: dict,
    enable_thinking: bool = False,
    max_tokens: int = 16384,
    messages: list | None = None,
    stop_words: list | None = None,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    sampling_seed: int = 0,
    count_tokens_only: bool = False,  # 新增参数
) -> dict:
    """
    Internal function to make API requests with shared logic.
    """
    base_url = (
        OPENAI_API_BASE_THINKING # must be thinking mode to get the model name
    )
    client = OpenAI(
        base_url=base_url,
        api_key=OPENAI_API_KEY,
    )
    models = client.models.list()
    model = models.data[0].id

    # # Prepare extra_body parameters
    # extra_body = extra_body.copy()  # Don't modify the original dict
    # extra_body["top_k"] = top_k

    # if enable_thinking: # must be thinking mode to get the model name
    #     if "chat_template_kwargs" not in extra_body:
    #         extra_body["chat_template_kwargs"] = {}
    #     extra_body["chat_template_kwargs"]["enable_thinking"] = True

    print(
        f"Requesting DeepSeek with enable_thinking={enable_thinking}, max_tokens={max_tokens}, temperature={temperature}, top_p={top_p}, top_k={top_k}, sampling_seed={sampling_seed}"
    )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        # top_p=top_p,
        stop=stop_words,
        seed=sampling_seed * 10,  # Use a different seed for each sampling
        # extra_body=extra_body,
    )

    reasoning_content = response.choices[0].message.reasoning_content
    content = response.choices[0].message.content
    
        
        

    # # If reasoning content is not found, return None
    # if reasoning_content is None:
    #     print("Warning: No reasoning content found in the response.")
    #     reasoning_content = ""
    # if "<think>" in reasoning_content:
    #     reasoning_content = reasoning_content.split("<think>")[1].strip()

    return {
        "reasoning_content": reasoning_content,
        "content": content,
    }


def _make_request(
    extra_body: dict,
    enable_thinking: bool = False,
    max_tokens: int = 16384,
    messages: list | None = None,
    stop_words: list | None = None,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    sampling_seed: int = 0,
    count_tokens_only: bool = False,  # 新增参数
) -> dict:
    """Function to make a request to the OpenAI API.

    Args:
        extra_body: Additional parameters for the request
        enable_thinking: If True, use thinking mode with different default parameters
        max_tokens: Maximum number of tokens to generate
        messages: List of messages to include in the request
        stop_words: List of stop words to use in the generation
        temperature: Temperature for sampling
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        sampling_seed: Seed for random sampling

    Returns:
        Response from the OpenAI API
    """
    if count_tokens_only:
        return _make_request_tokenize_count(
            extra_body=extra_body,
            enable_thinking=enable_thinking,
            max_tokens=max_tokens,
            messages=messages,
            stop_words=stop_words,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            sampling_seed=sampling_seed,
        )
    else:
        return _make_request_response(
            extra_body=extra_body,
            enable_thinking=enable_thinking,
            max_tokens=max_tokens,
            messages=messages,
            stop_words=stop_words,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            sampling_seed=sampling_seed,
        )


def request_qwq(
    extra_body: dict,
    thinking: bool = True,
    max_tokens: int = 16384,
    messages: list | None = None,
    stop_words: list | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int = 20,
    sampling_seed: int = 0,
) -> dict:
    """
    Function to request a response from the OpenAI API.

    Args:
        thinking: If True, use thinking mode with different default parameters
        temperature: If None, will use 0.6 for thinking mode, 0.7 for non-thinking
        top_p: If None, will use 0.95 for thinking mode, 0.8 for non-thinking
    """
    
    
    assert thinking, "QwQ model only supports thinking mode."
    
    # Set default parameters based on thinking mode
    temperature = 0.6 # must be thinking mode to get the model name
    top_p = 0.95 # must be thinking mode to get the model name

    print(
        f"Requesting qwq with thinking={thinking}, max_tokens={max_tokens}, temperature={temperature}, top_p={top_p}, top_k={top_k}, sampling_seed={sampling_seed}"
    )

    # calculate token count if messages are provided
    if messages is not None:
        token_count = _make_request(
            extra_body=extra_body,
            enable_thinking=thinking,
            max_tokens=max_tokens,
            messages=messages,
            stop_words=stop_words,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            sampling_seed=sampling_seed,
            count_tokens_only=True,
        )["token_count"]
        print(f"DEEPSEEK Token count: {token_count}")
    else:
        token_count = 0
        print("No messages provided, token count is 0.")

    if token_count + max_tokens > DEEPSEEK_MAX_TOKENS:
        print(
            f"Warning: The total token count ({token_count + max_tokens}) exceeds the maximum limit ({DEEPSEEK_MAX_TOKENS})."
        )
        max_tokens = DEEPSEEK_MAX_TOKENS - token_count - 200  # Leave some buffer

    count=1
    while True:
        try:
            results = _make_request(
                extra_body=extra_body,
                enable_thinking=thinking,
                max_tokens=max_tokens,
                messages=messages,
                stop_words=stop_words,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                sampling_seed=sampling_seed*count,
            )
            if results is not None and (("reasoning_content" in results and results["reasoning_content"] is not None) or ("content" in results and results["content"] is not None)):
                break  # Exit loop if request is successful
            else:
                print("Request returned None or missing fields, retrying...")
                count += 1
                if count > 5:
                    raise ValueError("Request failed after 5 attempts, please check the model or parameters.")
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}. Retrying...")
            count += 1

    return results
    # results = _make_request(
    #     extra_body=extra_body,
    #     enable_thinking=thinking,
    #     max_tokens=max_tokens,
    #     messages=messages,
    #     stop_words=stop_words,
    #     temperature=temperature,
    #     top_p=top_p,
    #     top_k=top_k,
    #     sampling_seed=sampling_seed,
    #     count_tokens_only=False,
    # )

    # if thinking and results["reasoning_content"] is None:
    #     print(
    #         "Warning: In reasoning mode, the reasoning content is None, retry with non reasoning mode."
    #     )
    #     return _make_request(
    #         extra_body=extra_body,
    #         enable_thinking=False,
    #         max_tokens=max_tokens,
    #         messages=messages,
    #         stop_words=stop_words,
    #         temperature=temperature,
    #         top_p=top_p,
    #         top_k=top_k,
    #         sampling_seed=sampling_seed,
    #     )
    # else:
    #     return results


if __name__ == "__main__":
    # # 示例用法展示新的 request_qwen 函数
    # # example of hard math question
    # messages = [
    #     {
    #         "role": "user",
    #         "content": "please explain the following math problem: 1+2+3+4+5+...+100",
    #     }
    # ]
    # extra_body = {}

    # # 使用默认thinking模式 (temperature=0.6, top_p=0.95)
    # print("Thinking模式 (默认参数):")
    # response = request_qwen(extra_body=extra_body, messages=messages)
    # print(response)

    # # # 使用非thinking模式 (temperature=0.7, top_p=0.8)
    # # print("Non-thinking模式 (默认参数):")
    # # response = request_qwen(extra_body=extra_body, thinking=False, messages=messages)
    # # print(response)

    # # 手动指定参数会覆盖默认值
    # print("Thinking模式 (自定义参数):")
    # # response = request_qwen(
    # #     extra_body=extra_body,
    # #     thinking=True,
    # #     temperature=0.8,
    # #     top_p=0.9,
    # #     messages=messages
    # # )

    pass
