# SCALE: Selective Resource Allocation for Overcoming Performance Bottlenecks in Mathematical Test-time Scaling

[![Paper](https://img.shields.io/badge/Paper-AAAI%202026-blue)](https://arxiv.org/abs/2512.00466)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-DualThinking-yellow)](https://huggingface.co/datasets/YangXiao-nlp/DualThinking)

Official implementation of **"SCALE: Selective Resource Allocation for Overcoming Performance Bottlenecks in Mathematical Test-time Scaling"** accepted at AAAI 2026.

📄 **Paper**: [SCALE: Selective Resource Allocation for Overcoming Performance Bottlenecks in Mathematical Test-time Scaling](https://arxiv.org/abs/2512.00466)  
🤗 **Dataset**: [YangXiao-nlp/DualThinking](https://huggingface.co/datasets/YangXiao-nlp/DualThinking)

---

## Abstract

This repository contains the official implementation of our AAAI 2026 paper on **SCALE** (Selective Resource Allocation), a novel three-stage adaptive reasoning framework for mathematical problem solving. Our approach dynamically selects between thinking and non-thinking modes based on step-level difficulty assessment, achieving significant improvements in computational efficiency while maintaining high accuracy.

## Overview

The SCALE (Selective Resource Allocation) system implements a four-stage mathematical reasoning pipeline:

1. **Problem Decomposition**: Generates multiple candidate decompositions and selects the most coherent step-by-step outline via majority voting
2. **Difficulty Assessment**: Assigns a difficulty score in \([0, 1]\) to each sub-problem conditioned on accumulated context
3. **Adaptive Mode Selection**: Chooses between fast System 1 processing and deliberate System 2 reasoning based on a user-configurable threshold
4. **Sequential Execution with Context Propagation**: Solves sub-problems in order while propagating intermediate results to maintain a coherent reasoning chain

### Key Features

- **Adaptive Mode Selection**: Dynamically chooses between thinking and non-thinking modes based on step difficulty
- **Majority Voting**: Generates multiple outline versions and selects the optimal one
- **Extensible Architecture**: Base class design supports easy integration of different language models
- **Incremental Processing**: Supports checkpointing and resumption of long-running inference tasks

## System Architecture

The core implementation is in [`inference/cot_dual_thinking.py`](inference/cot_dual_thinking.py), which provides:

```text
System3Base (Base Class)
├── thinking_outline_majority_voting()    # Stage 1: Problem decomposition via majority voting
├── thinking_scores()                      # Stage 2: Difficulty assessment for each sub-problem
├── thinking_details()                     # Stages 3-4: Adaptive mode selection and sequential execution
└── extract_final_answer()                 # Extract and validate final answer
```

### Class Hierarchy

```text
System3Base (cot_dual_thinking.py)
├── System3Distill (cot_dual_thinking_distill.py)  # DeepSeek Distill variant
└── System3QwQ (cot_dual_thinking_qwq.py)          # QwQ model variant
```

Subclasses override `get_thinking_request_func()` and `get_non_thinking_request_func()` to use different language models.

## Installation

### Prerequisites

- Python 3.8+
- OpenAI-compatible API endpoint (vLLM recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/DualThinking.git
cd DualThinking

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```text
openai>=1.40.0
requests>=2.31.0
```

## Configuration

Set the following environment variables for model endpoints:

```bash
# For Qwen models (default)
export QWEN_THINKING_BASE="http://localhost:8000/v1"
export QWEN_NONTHINKING_BASE="http://localhost:8000/v1"
export QWEN_OPENAI_API_KEY="your-api-key"

# For DeepSeek Distill variant
export DEEPSEEK_BASE="http://localhost:8001/v1"
export DEEPSEEK_TOKENIZE="http://localhost:8001/tokenize"

# For QwQ variant
export QWQ_BASE="http://localhost:8002/v1"
```

## Usage

### Basic Usage

```python
from inference.cot_dual_thinking import inference_sysytem3
import argparse

# Configure arguments
args = argparse.Namespace(
    output_folder="./output",
    max_tokens=32768,
    top_k=20,
    difficulty=0.5,  # Difficulty threshold for thinking mode
    cot_name="dual_thinking",
    model_name="qwen",
    prompt_version="v1"
)

# Prepare question
question = {
    "id": "math_001",
    "sampling_id": 0,
    "problem": "Your mathematical problem here",
    "condition": "Given conditions",
    "question": "What to find?",
    "answer": "Expected answer"
}

# Run inference
success = inference_sysytem3(question, args)
```

### Command-Line Interface

```bash
# Example: Run dual thinking inference on a dataset
python -m inference.cot_dual_thinking \
    --input data/eval/aime24/test_converted.jsonl \
    --output ./output \
    --max_tokens 32768 \
    --difficulty 0.5 \
    --model_name qwen
```

### Advanced: Using Variants

```python
# Using DeepSeek Distill variant
from inference.cot_dual_thinking_distill import System3Distill

system3 = System3Distill(question, args)
# ... run inference

# Using QwQ variant
from inference.cot_dual_thinking_qwq import System3QwQ

system3 = System3QwQ(question, args)
# ... run inference
```

## Project Structure

```text
DualThinking/
├── inference/
│   ├── cot_dual_thinking.py          # Core implementation (System3Base)
│   ├── cot_dual_thinking_distill.py  # DeepSeek Distill variant
│   ├── cot_dual_thinking_qwq.py      # QwQ model variant
│   ├── qwen.py                        # Qwen model interface
│   ├── deepseek_distill.py           # DeepSeek Distill interface
│   ├── qwq.py                         # QwQ model interface
│   ├── util.py                        # Utility functions
│   └── convert_problem_format.py     # Data preprocessing
├── data/
│   ├── eval/                          # Evaluation datasets
│   │   ├── aime24/                   # AIME 2024 test set
│   │   └── amc23/                    # AMC 2023 test set
│   └── train/                         # SCALE-generated reasoning traces
│       ├── limo/                     # Traces from LIMOPro dataset using QwQ
│       └── limo_v2/                  # Enhanced version of LIMOPro traces
├── paper/
│   └── AAAI26.pdf                    # Paper PDF
├── requirements.txt                   # Python dependencies
└── README.md                         # This file
```

## Algorithm Details

### Stage 1: Problem Decomposition (Majority Voting)

SCALE samples eight alternative outlines for each problem and selects the most coherent version via a self-evaluation prompt.

```python
def thinking_outline_majority_voting(self) -> None:
    outlines = {}
    for version_id in range(1, self.step_outline_versions + 1):
        version = self.thinking_outline_single(generate_id=version_id * sampling_id)
        outlines[f"version{version_id}"] = json.loads(version)

    selected_version = evaluate_and_select(outlines, problem)
    self.steps = outlines[selected_version]
```

This mirrors the paper's decomposition module by producing an ordered reasoning outline before any detailed solving begins.

### Stage 2: Difficulty Assessment

Each sub-problem receives a difficulty score in \([0, 1]\) conditioned on the accumulated solution context:

- Scores near **0** denote routine operations.
- Scores near **1** denote challenging reasoning steps.

The difficulty threshold is user-configurable through `--difficulty` (default: 0.5), allowing practitioners to tune how selectively SCALE engages deliberate reasoning.

### Stage 3: Adaptive Mode Selection

SCALE compares each difficulty score against the threshold to decide whether to invoke fast System 1 processing or deliberate System 2 reasoning.

```python
enable_thinking = self.steps_scores[step] > self.threshold
request_func = (
    self.get_thinking_request_func()
    if enable_thinking
    else self.get_non_thinking_request_func()
)
```

Lower thresholds (e.g., `--difficulty 0.3`) encourage broader use of System 2, while higher thresholds (e.g., `--difficulty 0.7`) reserve it for the most demanding sub-problems.

### Stage 4: Sequential Execution with Context Propagation

Reasoning proceeds step-by-step while propagating intermediate results so that downstream sub-problems retain full contextual knowledge.

```python
for step_idx, step in enumerate(self.steps, start=1):
    enable_thinking = self.steps_scores[step] > self.threshold
    request_func = (
        self.get_thinking_request_func()
        if enable_thinking
        else self.get_non_thinking_request_func()
    )

    current_process = " ".join(
        f"{self.steps[f'step{i}']}\nstep answer:{self.steps_content[f'step{i}']}"
        for i in range(1, step_idx)
    )

    query = f"Given conditions: {conditions}\nCurrent process: {current_process}\nQuestion: {step}"
    results = request_func(thinking=enable_thinking, messages=messages, ...)
    self.steps_detail[f"step{step_idx}"] = results["reasoning_content"]
    self.steps_content[f"step{step_idx}"] = results["content"]
```

This stage ensures the final answer is derived from a coherent reasoning chain that accumulates both sub-problems and their solutions.

## Training Data Generation

The training data consists of high-quality synthetic reasoning traces generated using our SCALE framework:

1. **Source**: Problems from the LIMOPro (Xiao et al. 2025) dataset
2. **Generation**: Applied SCALE framework on QwQ model to synthesize reasoning traces
3. **Filtering**: Removed instances where SCALE-generated answers differ from original LIMOPro answers
4. **Result**: 800 high-quality question-response pairs with SCALE-generated reasoning traces

This synthetic data is used for supervised fine-tuning of base models (Qwen2.5-14B/32B/72B-Instruct, Llama3.3-70B-Instruct) to enhance non-reasoning model performance.

## Evaluation

The system has been evaluated on the following benchmarks:

- **AIME 2024**: American Invitational Mathematics Examination
- **AMC 2023**: American Mathematics Competition

Dataset and evaluation data are available at: [🤗 YangXiao-nlp/DualThinking](https://huggingface.co/datasets/YangXiao-nlp/DualThinking)

Results and detailed analysis are available in our [paper](paper/AAAI26.pdf).

## Output Format

The system saves results in JSON format with the following structure:

```json
{
    "id": "math_001",
    "sampling_id": 0,
    "problem": "Original problem text",
    "steps": {
        "step1": "First subproblem",
        "step2": "Second subproblem",
        ...
    },
    "steps_scores": {
        "step1": 0.3,
        "step2": 0.7,
        ...
    },
    "steps_detail": {
        "step1": "Detailed reasoning for step 1",
        ...
    },
    "steps_content": {
        "step1": "Solution for step 1",
        ...
    },
    "final_answer": "42",
    "is_correct": true,
    "cot_name": "dual_thinking",
    "model_name": "qwen"
}
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_tokens` | 32768 | Maximum tokens for response generation |
| `difficulty` | 0.5 | Difficulty threshold for thinking mode activation (0.0-1.0). Steps with scores > threshold use thinking mode. Adjust based on resource constraints: lower values (e.g., 0.3) use thinking mode more aggressively; higher values (e.g., 0.7) use it more conservatively. |
| `top_k` | 20 | Top-k sampling parameter |
| `step_outline_versions` | 8 | Number of outline versions for majority voting |

## Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{xiao2025scaleselectiveresourceallocation,
      title={SCALE: Selective Resource Allocation for Overcoming Performance Bottlenecks in Mathematical Test-time Scaling}, 
      author={Yang Xiao and Chunpu Xu and Ruifeng Yuan and Jiashuo Wang and Wenjie Li and Pengfei Liu},
      year={2025},
      eprint={2512.00466},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.00466}, 
}
```

Please also cite the LIMOPro dataset which provides the source problems for our training data:

```bibtex
@article{xiao2025limopro,
  title={LIMOPro: Reasoning Refinement for Efficient and Effective Test-time Scaling},
  author={Xiao, Yang and Wang, Jiashuo and Yuan, Ruifeng and Xu, Chunpu and Xu, Kaishuai and Li, Wenjie and Liu, Pengfei},
  journal={arXiv preprint arXiv:2505.19187},
  year={2025}
}
```

## Dataset

Our training and evaluation datasets are publicly available on Hugging Face:

🤗 [YangXiao-nlp/DualThinking](https://huggingface.co/datasets/YangXiao-nlp/DualThinking)

The dataset includes:

- **Training data**: High-quality synthetic reasoning traces generated by applying the SCALE framework on QwQ model using problems from LIMOPro dataset (800 curated question-response pairs)
- **Evaluation sets**: AIME 2024 and AMC 2023 test sets for benchmark evaluation
- **Converted formats**: Preprocessed data formats for easy integration with the inference pipeline

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions and feedback, please open an issue on GitHub or contact the authors.

## Acknowledgments

We thank the AAAI 2026 reviewers for their valuable feedback and suggestions.
