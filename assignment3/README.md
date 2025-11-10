# COMP4901B Homework 3 — Self-Improvement for Math Reasoning

**Due Date: Nov 20, 2025, 11:59 PM**

**Full score: 100 points.**

This repository contains the starter code and tooling for Homework 3. You will:
- Implement VLLM-based inference to generate multiple solution rollouts for math problems
- Implement answer extraction and verification logic to filter correct solutions
- Implement LoRA (Low-Rank Adaptation) module selection for parameter-efficient fine-tuning
- Perform iterative self-training on the GSM8K benchmark to improve model performance
- Analyze the impact of self-improvement across multiple training iterations

You can use your own laptop/machine if it has a GPU, the GPU cluster provided by HKUST CSE, or cloud platforms (e.g., Google Colab) to perform the experiments.

This writeup is organized as follows:
- [Rules](#rules) contains the base rules
- [Your Tasks](#your-tasks) is the main part and contains all the tasks you should complete for this homework
- [Background](#background) provides essential context about self-training and GSM8K
- [Setup](#setup) describes the basic setup for this codebase
- [Important Notes](#important-notes) contains notes about using the code
- [Submission](#submission) contains instructions on how to submit your homework
- [Quick Reference](#quick-reference) provides a quick command reference
- [Repo Structure](#repo-structure) describes the repository organization
- [Troubleshooting](#troubleshooting) contains helpful tips

## Rules

Detailed course logistics is at [here](https://docs.google.com/document/d/1mWm_TYYQpD3NpJISlFQGurBIXxWjizEc1zVffIDhiXU/edit?usp=sharing).

1. You need to work on the homework independently, without collaborating with other humans.
2. You are allowed to collaborate with other AIs. You are encouraged to use AI tools to help explain unfamiliar parameters or tools, as we cannot cover everything in lectures.
3. As noted in the logistics, each student will have a total of three free late (calendar) days to use for homeworks. Once these late days are exhausted, any assignments turned in late will be penalized 20% per late day. However, no assignment will be accepted more than three days after its due date.
4. For questions about this homework, please post on Canvas.

Honor code violation will directly cause failing the course.

## Your Tasks

In this homework, the main goal is to implement a self-improvement recipe that we have learned in the lectures. Basically the model first generates data from its own for a mathematical reasoning task, then we filter the generated data with final answer correctness, and fine-tune the model on the filtered data.

### 0. Environment Setup

- **We recommend running the project on a Linux system. Windows and macOS environments are not tested and may not work as expected.**
- Run `bash scripts/setup.sh` from the repo root to install all dependencies and download necessary resources.
- The script installs PyTorch, Transformers, VLLM, PEFT, and other required packages.
- It also downloads the GSM8K dataset and sets up the evaluation environment.

### 1. (15 Points) Part 1 — VLLM Inference Implementation

Implement the inference logic in `inference_vllm.py` to generate model outputs for GSM8K problems.

**Task Details:**
- Complete the TODO in the `format_prompts()` function to format GSM8K questions into prompts.
  - Support both zero-shot and few-shot (8 examples) prompting modes
  - Apply chat templates when using instruction-tuned models
  - Handle system messages and special tokens correctly

- Complete the TODO in the `run_inference()` function to perform VLLM-based generation.
  - Initialize the VLLM engine with appropriate parameters
  - Configure sampling parameters (temperature, top-p, top-k)
  - **Important**: Understand the difference between greedy decoding (temperature=0.0) and sampling-based generation (temperature>0.0)
  - Support multiple rollouts per question for diversity

**Validation:**
- Run the sanity check to validate your implementation:
  ```bash
  python tests/test_format_prompts.py
  ```
- Note: This test requires VLLM to be installed. If not available, you can manually verify your prompt formatting by printing sample outputs.

**Report (paste into the PDF):**
- Screenshot or console output showing that your implementation works correctly
- Brief explanation of your implementation approach
- **Answer the question**: Why do we need temperature > 0 when generating multiple rollouts? What would happen if we used temperature = 0 for multiple rollouts?

### 2. (20 Points) Part 2 — Answer Verification Implementation

Implement the answer extraction and scoring logic in `evaluate_model_outputs.py`.

**Task Details:**

**2.1 `extract_solution()` function:**
- Extract numerical answers from model-generated solutions
- Handle LaTeX `\boxed{}` format: e.g., "The answer is \boxed{42}"
- Handle plain text format: e.g., "Therefore, the result is 42."
- Return None if no answer is found
- Clean up formatting (remove commas, dollar signs, etc.)

**2.2 `compute_score()` function:**
- Compare extracted answer with ground truth
- Normalize both answers (handle "42" vs "42.0")
- Return 1 for correct, 0 for incorrect
- Handle edge cases (None answers, non-numeric values)

**2.3 `estimate_pass_at_k()` function:**
- Implement the pass@k metric estimator
- **Challenge**: Derive the mathematical formula yourself
- **Conceptual understanding**: If you have n samples and c are correct, what is the probability that at least one of k randomly selected samples is correct?
- **Hint**: Think about the complement probability: P(at least 1 correct) = 1 - P(all k are wrong)
- **Hint**: Consider combinatorics - how many ways can you choose k items from n items?
- Handle numerical stability (avoid computing large factorials directly)
- Support both uniform and per-problem sample counts

**Background on pass@k:**

The pass@k metric measures the probability that at least one correct solution appears when randomly sampling k solutions from n generated samples. Mathematically, if we have n total samples where c are correct:

- Total samples: n
- Correct samples: c  
- Incorrect samples: n - c
- Samples to select: k (without replacement)

The pass@k probability is computed as:

**pass@k = P(at least 1 correct among k samples)**

**Key insights for derivation:**
- **Hint 1**: Use the complement probability approach:  
  P(at least 1 correct) = 1 - P(all k selected are wrong)

- **Hint 2**: Think about combinatorics. The number of ways to choose k items from a set of n items is the binomial coefficient: C(n, k) = n! / (k! × (n-k)!)

- **Hint 3**: For numerical stability with large numbers, avoid computing factorials directly. Instead, use a product formulation. You can compute C(n-c, k)/C(n, k) as a product of ratios.

Pass@k is an important metric to understand how many samples we need for one question to obtain at least one correct solution. This is important for data synthesis.


**Validation:**
Run the sanity check to validate all three functions:
```bash
python tests/test_evaluation_functions.py
```

The checker will test your implementations on multiple test cases with known reference values.

**Report (paste into the PDF):**
- Screenshot showing all test cases passed
- Brief explanation of your extraction and scoring logic
- **Derivation**: Show your mathematical derivation of the pass@k formula (include equations)
- Explanation of how you handled numerical stability in the pass@k computation

### 3. (25 Points) Part 3 — LoRA Training Implementation

Implement the LoRA adapter application logic in `train_gsm8k_self_training_lora.py`.

**Task Details:**

**3.1 `apply_adapters()` method in `LoRAAdapterManager` class:**
- Apply the LoRA configuration to the model
- Use the appropriate function from the PEFT library
- **Hint**: Check the imports at the top of the file

**3.2 `_resolve_lora_target_modules()` method in `LoRAAdapterManager` class:**
- Automatically detect which linear layers to apply LoRA to
- **Your task**:
  1. Find all linear layers (nn.Linear modules) in the model
  2. Detect the model architecture type (e.g., "llama", "opt", "mistral", "qwen")
  3. Select appropriate target modules based on architecture. The architecture of qwen3 could be refered at [Qwen3-HF](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py)
  4. Validate that selected modules exist in the model
  5. Return a list of module names (e.g., ["q_proj", "k_proj", "v_proj", "o_proj"])

- **Hint**: Different architectures use different naming conventions:
  - Attention layers: q_proj, k_proj, v_proj, o_proj (or out_proj)
  - FFN layers: gate_proj, up_proj, down_proj, fc1, fc2
- **Hint**: Use `self.model.named_modules()` to iterate through all modules
- **Hint**: Use `self.model.config.model_type` to detect the architecture
- **Hint**: If user specified `--lora_target_modules` via command line, use those instead

**Note**: There is no automatic sanity check for this part. You will validate correctness by successfully running training in Part 4.

**Report (paste into the PDF):**
- Code snippet showing your implementation of `_resolve_lora_target_modules()`
- Explanation of how you detected the model architecture and selected target modules
- **Answer the question**: Why is it beneficial to apply LoRA to both attention and FFN layers? What would happen if we only applied LoRA to attention layers?

### 4. (10 Points) Part 4 — Self-Training Execution

Perform iterative self-training on the GSM8K dataset using the provided pipeline.

**Task Details:**
- Use the script `scripts/self_train_gsm8k.sh` to launch the self-training loop
- The script performs multiple iterations of:
  1. **Inference**: Generate multiple solution rollouts per question
  2. **Verification**: Filter correct solutions using your implemented functions
  3. **Training**: Fine-tune the model on correct solutions using LoRA
- Default settings: 1 iterations, 8 rollouts per question, temperature 1.0, LoRA rank 64
- Training will use the filtered correct solutions from each iteration

**Running Self-Training:**
```bash
cd /path/to/COMP4901B-Homework3
export WANDB_API_KEY="YOUR_WANDB_API_KEY"

export WANDB_PROJECT="COMP4901B-Homework3"
export CUDA_VISIBLE_DEVICES="0"


bash scripts/self_train_gsm8k.sh \
    Qwen3-0.6B 
```

**Key Configuration** (in `scripts/self_train_gsm8k.sh`):
- `BASE_MODEL`: Specifies the base model to be used. You should provide the model path (e.g., "Qwen3-0.6B").
- `NUM_ITERATIONS`: Number of self-training iterations (default: 1)
- `N_ROLLOUTS`: Number of rollouts per question during inference (default: 8)
- `TEMPERATURE`: Sampling temperature (default: 1.0)
- `LORA_R`: LoRA rank (default: 64)
- `LEARNING_RATE`: Learning rate for LoRA training (default: 2e-5)
- `NUM_EPOCHS`: Number of training epochs per iteration (default: 1)
- `N_QUERIES`: Number of queries to be used for self-training (default: 2000)

With the default config, the training takes about 5 hours on 2080ti. 

**The trained model will be saved in the directory `ckpt/RUN_NAME/models/model_iter_xxx-merged`**

**Report (paste into the PDF):**
- Training configuration summary (model, iterations, rollouts, LoRA config)
- Table showing metrics across iterations:
  - Number of correct solutions found
  - pass@1, pass@4, pass@8 scores
  - Training loss
- Screenshot or plot of accuracy improvement across iterations
- Analysis: Did self-training improve performance? Why or why not?

**Hints for Improvement**

- To achieve better performance, consider experimenting with a different learning rate or batch size. For LoRA, you can often use a larger learning rate than what is typically used for full fine-tuning.
- Increasing the N_QUERIES for training may also lead to better results, though it will extend the total training time.

### 5. (30 Points) Part 5 — Final Evaluation and Analysis

Evaluate your final model and analyze the self-training process.

**Task Details:**
- Run evaluation on the GSM8K test set using the script `scripts/run_gsm8k_eval.sh`
- Compare performance between:
  1. Base model (iteration 0)
  2. After iteration 1

**Running Evaluation:**
```bash
cd /path/to/COMP4901B-Homework3

# Evaluate base model
bash scripts/run_gsm8k_eval.sh \
    --model_path Qwen3-0.6B  \
    --output_dir results/baseline

# Evaluate fine-tuned model (replace path with your checkpoint)
bash scripts/run_gsm8k_eval.sh \
    --model_path PATH_TO_THE_MERGED_MODLE \
    --output_dir results/xxxxx

# Try multiple rollout evaluation for both base and trained model
bash run_gsm8k_eval.sh \
    YOUR_MODEL_PATH \
    --output_dir results/xxxxx \
    --temperature 0.6 \
    --max_tokens 512 \
    --top_p 0.95 \
    --top_k 20 \
    --n_rollouts 8 \
    --n_queries 1000
```

**Scoring Rubric:**

**Baseline Evaluation (5 points):**
- Students must achieve at least **57% score** on the baseline (Qwen3-0.6B) to receive full 5 points
- If baseline score is below 57, students should check their Part 1 & 2 implementation
- Points are awarded proportionally based on baseline accuracy

**Trained Model Evaluation (20 points):**
- Students must achieve at least **1 point improvement** over baseline after training to receive 10 points
- Additional 10 points for comprehensive analysis with clear insights (as requested below), ablation studies (e.g. you may try reporting the performance of different configurations that you have tried and how they affect the performance), or higher accuracy

**Pass@k Analysis (5 points):**
- Evaluate baseline and trained models with multiple rollouts (n_rollouts=8)
- Report avg@k and pass@k metrics for k = 1, 2,..., 8
- Analyze the trend: how do avg@k and pass@k change as k increases?
- Compare these metric before and after training ? Do they increase ? Why ? 
- Explain what these trends indicate about model diversity and reliability

**Report (paste into the PDF):**
- **Baseline Evaluation**:
  - Screenshot of baseline model evaluation results
  - Report baseline accuracy score (must be ≥57)

- **Trained Model Evaluation**:
  - Screenshot of trained model evaluation results
  - Comparison table showing accuracy across all iterations
  - Report improvement over baseline (must be ≥1 point)

- **Pass@k and Avg@k Analysis**:
  - Screenshot of multiple rollout evaluation (for both baseline and trained models)
  - Table showing avg@1, avg@4, avg@8 for baseline and trained models
  - Table showing pass@1, pass@4, pass@8 for baseline and trained models
  - **Trend analysis**: Analyze how avg@k and pass@k change as k increases (1→4→8)
    - What does the trend tell you about solution diversity?
    - Does the trained model show better diversity than baseline?
    - For baseline, what does the gap between avg@k and pass@k indicate? 


## More Background

### What is Self-Training for Math Reasoning?

Self-training (also called self-improvement) is a technique where a model generates its own training data by:
1. Generating multiple candidate solutions for each problem
2. Verifying which solutions are correct
3. Training on the correct solutions to improve performance

This approach is particularly effective for math reasoning because:
- We can automatically verify correctness (numerical answers)
- Multiple solution paths often exist for the same problem
- Models can learn from their own diverse reasoning strategies

### The Self-Training Loop

```
Iteration 0: Base model
    ↓
Generate multiple solutions (with sampling)
    ↓
Verify solutions (extract & check answers)
    ↓
Filter correct solutions
    ↓
Fine-tune model on correct solutions (LoRA)
    ↓
Iteration 1: Improved model
    ↓
[Repeat...]
```


### Pass@k Metric

The pass@k metric measures: "Given k solution attempts, what's the probability that at least one is correct?"

This is more informative than simple accuracy when we can generate multiple solutions:
- pass@1: Traditional accuracy (single attempt)
- pass@4: Success rate with 4 attempts
- pass@8: Success rate with 8 attempts

Higher pass@k with the same pass@1 means the model has diverse reasoning strategies.

### LoRA (Low-Rank Adaptation)

LoRA is a parameter-efficient fine-tuning method that:
- Freezes the original model weights
- Adds small trainable low-rank matrices to specific layers
- Achieves comparable performance to full fine-tuning with <1% trainable parameters
- Enables fast training and easy model merging

**Key hyperparameters**:
- **r (rank)**: Size of low-rank matrices (typical: 8-128)
- **alpha**: Scaling factor for LoRA updates (typical: 16-32)
- **target_modules**: Which layers to apply LoRA to (e.g., attention projections)

### GSM8K Dataset

[GSM8K](https://huggingface.co/datasets/gsm8k) is a dataset of 8.5K grade school math word problems requiring multi-step reasoning.

**Example:**
```
Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast
every morning and bakes muffins for her friends every day with four. She
sells the remainder at the farmers' market daily for $2 per fresh duck egg.
How much in dollars does she make every day at the farmers' market?

Answer: Janet sells 16 - 3 - 4 = 9 duck eggs every day.
She makes 9 * 2 = $18 every day at the farmer's market.
#### 18
```

The answer format is: `reasoning #### numerical_answer`

## Setup

### Installation

Run the setup script to install all dependencies:
```bash
bash scripts/setup.sh
```

The script will:
- Install PyTorch 2.8.0 with CUDA 12.6 support
- Install Transformers, VLLM, PEFT, and other required packages
- Download the GSM8K dataset


## Important Notes

### Sanity Checks

Before running the full pipeline, validate your implementations using the provided tests:

```bash
# Test format_prompts (requires VLLM)
python tests/test_format_prompts.py

# Test evaluation functions (no VLLM required)
python tests/test_evaluation_functions.py

# Run all tests
python tests/run_all_tests.py
```

**Important**: The sanity checks use pre-computed reference outputs to verify correctness. Make sure all tests pass before proceeding to training.

### Data Format

**Inference Output Format** (`inference_vllm.py`):
```json
{
  "idx": 0,
  "question": "Janet's ducks lay 16 eggs...",
  "input": "<formatted prompt with chat template>",
  "gold_answer": "Janet sells 16 - 3 - 4 = 9... #### 18",
  "gt": "18",
  "model_output": "<model generated solution>"
}
```

**Evaluation Output Format** (`evaluate_model_outputs.py`):
```json
{
  "idx": 0,
  "question": "Janet's ducks lay 16 eggs...",
  "input": "<formatted prompt>",
  "gold_answer": "Janet sells 16 - 3 - 4 = 9... #### 18",
  "gt": "18",
  "model_output": "<model generated solution>",
  "extracted_answer": "18",
  "score": 1
}
```

### VLLM Configuration

For inference, you can adjust VLLM settings in `inference_vllm.py`:

- `--tensor_parallel_size`: Number of GPUs for model parallelism
- `--gpu_memory_utilization`: Fraction of GPU memory to use (default: 0.8)
- `--max_tokens`: Maximum tokens to generate per completion
- `--n_rollouts`: Number of solutions to generate per question



## Submission

Submit a zip of the codebase (only the homework directory) and a PDF report to Canvas. Your PDF should include your full name, student ID.

**Important**: Please submit the codebase without any model checkpoints and please submit the codebase zip and PDF report in **two** seperate files. 

## Quick Reference

```bash
# Setup
bash scripts/setup.sh

# Sanity checks (Parts 1 & 2)
python tests/test_format_prompts.py              # Part 1
python tests/test_evaluation_functions.py        # Part 2
python tests/run_all_tests.py                    # All tests

# Self-training (Parts 3 & 4)
bash scripts/self_train_gsm8k.sh \
    Qwen3-0.6B 

# Evaluation (Part 5)
bash scripts/run_gsm8k_eval.sh \
    --model_path <model_path> \
    --output_dir <output_dir>

```

## Repo Structure

```
.
├── inference_vllm.py                    # TODO: implement inference logic (Part 1)
├── inference_vllm_ref.py                # Reference implementation (do not modify)
├── evaluate_model_outputs.py            # TODO: implement verification (Part 2)
├── evaluate_model_outputs_ref.py        # Reference implementation (do not modify)
├── train_gsm8k_self_training_lora.py    # TODO: implement LoRA logic (Part 3)
├── train_gsm8k_self_training_lora_ref.py # Reference implementation (do not modify)
├── loss_functions.py                    # Loss computation (provided)
├── scripts/
│   ├── setup.sh                        # Environment setup
│   ├── self_train_gsm8k.sh            # Main self-training loop (Parts 3 & 4)
│   └── run_gsm8k_eval.sh              # Evaluation script (Part 5)
├── tests/
│   ├── test_format_prompts.py         # Sanity check for Part 1
│   ├── test_evaluation_functions.py   # Sanity check for Part 2
│   ├── run_all_tests.py               # Run all sanity checks
│   └── README.md                      # Testing documentation
├── eval_gsm8k-main/                    # GSM8K evaluation utilities
└── README.md                           # This file
```

## Troubleshooting

**1. "NotImplementedError" in inference_vllm.py**
- Complete the TODOs in `format_prompts()` and `run_inference()`
- Check the detailed comments in the code for guidance

**2. "NotImplementedError" in evaluate_model_outputs.py**
- Complete the TODOs in `extract_solution()`, `compute_score()`, and `estimate_pass_at_k()`
- Run `python tests/test_evaluation_functions.py` to validate

**3. "NotImplementedError" in train_gsm8k_self_training_lora.py**
- Complete the TODOs in `apply_adapters()` and `_resolve_lora_target_modules()`
- Check the imports and hints in the code

**4. "No module named 'vllm'"**
- Run `bash scripts/setup.sh` to install dependencies
- Or: `pip install vllm transformers peft`

**5. "CUDA out of memory" during inference**
- Reduce `--gpu_memory_utilization`  


**6. "CUDA out of memory" during training**
- Reduce batch size in training script
- Reduce `--lora_r` (e.g., 32 instead of 64)
- Enable gradient checkpointing

**7. Sanity check fails for estimate_pass_at_k**
- Make sure you're computing the complement probability correctly
- Check numerical stability (use product form, not factorials)
- Handle edge cases (n-c < k should return 1.0)
- Verify you're returning a numpy array

**8. Few or no correct solutions found during self-training**
- Check your `extract_solution()` implementation (are you extracting answers correctly?)
- Check your `compute_score()` implementation (are you comparing correctly?)
- Manually inspect some outputs to debug

**9. Training loss is NaN or not decreasing**
- Check that your LoRA modules were applied correctly
- Verify training data is in the correct format
- Try lower learning rate (2e-5 or 1e-5)


For more help:
- Post questions on Canvas
- Review code comments in source files
- Use AI tools to help understand concepts

