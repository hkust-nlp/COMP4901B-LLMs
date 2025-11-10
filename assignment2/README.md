# COMP4901B Homework 2 — Supervised Fine-Tuning (SFT) for Language Models

**Due Date: Nov 5, 2025, 11:59 PM**

**Full score: 100 points.**

This repository contains the starter code and tooling for Homework 2. You will:
- Implement loss masking logic for single-turn and multi-turn conversational data preprocessing
- Implement token-level cross-entropy loss for language modeling
- Perform supervised fine-tuning (SFT) on a small language model using conversational data
- Evaluate instruction-following capabilities using the IFEval benchmark
- Tune hyperparameters to achieve better performance
- Analyze the impact of SFT on model behavior and performance

You can use your own laptop/machine if it has a GPU, the 2080Ti GPU provided by HKUST CSE ([Usage Guide](school_cluster.md)), or free Google Colab ([Getting Started With Google Colab: A Beginners Guide](https://www.marqo.ai/blog/getting-started-with-google-colab-a-beginners-guide)) to perform the experiments.

This writeup is organized as follows:
- [Rules](#rules) contains the base rules
- [Your Tasks](#your-tasks) is the main part and contains all the tasks you should complete for this homework
- [Background](#background) provides essential context about SFT and loss masking
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

### 0. Environment Setup
<<<<<<< HEAD
=======
- **We recommend running the project on a Linux system. Windows and macOS environments are not tested and may not work as expected.**
>>>>>>> 846ec3382b0c09ac1adc60e65c032bec5399a431
- We provide access to 2080 Ti GPUs through the department cluster. You can refer to [this document](school_cluster.md) for detailed usage instructions.
- Run `bash setup.sh` from the repo root to install all dependencies and download necessary resources.
- The script installs PyTorch, Transformers, DeepSpeed, and other required packages.
- It also downloads the SmolLM2-135M model and the smol-smoltalk-6k dataset (a 6000-sample subset of the original smol-smoltalk dataset).
- In case you use Google colab or other machines, you can follow similar instructions.

<<<<<<< HEAD
### 1. (15 Points) Part 1 — Single-turn Loss Masking
=======
### 1. (5 Points) Part 1 — Single-turn Loss Masking
>>>>>>> 846ec3382b0c09ac1adc60e65c032bec5399a431

Implement the single-turn conversation loss masking logic in `conversation_func.py`.

**Task Details:**
- Complete the TODO in the `conversation_to_features()` function for single-turn conversations (Exercise 1).
- A single-turn conversation consists of (optionally) one system message, one user message, and one assistant message.
- **Goal:** Create a loss mask where only the assistant's response contributes to the training loss, while system and user tokens are masked out (set to `IGNORE_TOKEN_ID = -100`).
- Use `full_ids` (the complete tokenized conversation) and `prefix_lengths` (cumulative token counts per message) to identify the span corresponding to the assistant's response.
- Set `labels` to be a list where assistant tokens are copied from `full_ids` and all other positions are `IGNORE_TOKEN_ID`.
- Set `attention_mask` to be a list of 1s for all valid tokens (before padding).

**Validation:**
- Run `bash scripts/check_exercises_1.sh` (or `python conversation_func.py`) to validate your implementation against golden answers.
- The script will report whether your implementation matches the expected output.

**Report (paste into the PDF):**
- A screenshot or console output showing that your single-turn implementation passed validation.
- Brief explanation of your loss masking logic.

<<<<<<< HEAD
### 2. (15 Points) Part 2 — Multi-turn Loss Masking
=======
### 2. (5 Points) Part 2 — Multi-turn Loss Masking
>>>>>>> 846ec3382b0c09ac1adc60e65c032bec5399a431

Extend your implementation to support multi-turn conversations in `conversation_func.py`.

**Task Details:**
- Complete the TODO for multi-turn conversations (Exercise 2).
- Multi-turn conversations contain multiple (user, assistant) pairs, potentially preceded by system messages.
- **Goal:** Mask all user and system tokens with `IGNORE_TOKEN_ID`, and only keep assistant tokens for loss calculation.
- Handle truncation: when conversations exceed `max_length`, `prefix_lengths` may overshoot `len(full_ids)`. Use `min()` to stay in bounds.
- Iterate through all messages and identify which spans correspond to assistant utterances.

**Validation:**
- Run `bash scripts/check_exercises_2.sh` (or `python conversation_func.py --multi-turn`) to validate.
- Ensure your implementation handles conversations with arbitrary numbers of turns.

**Report (paste into the PDF):**
- A screenshot showing that your multi-turn implementation passed validation.
- Explanation of how you extended the single-turn logic to handle multiple turns.

<<<<<<< HEAD
### 3. (15 Points) Part 3 — Implementing Cross-Entropy Loss
=======
### 3. (5 Points) Part 3 — Reverse Loss Masking (Single-turn with Message Reordering)

Implement the reverse loss masking logic in `reverse_conversation_func.py` for single-turn conversations.

**Task Details:**
- Complete the TODO in the `reverse_conversation_to_features()` function for single-turn conversations (Exercise 3).
- **Key Mechanism**: The function **automatically reorders the messages** - putting the assistant message first and user message second:
  - Original: `[user: "..."] [assistant: "..."]`
  - After reordering: `[assistant: "..."] [user: "..."]`
  - Note: The messages are reordered, but their roles and content remain unchanged!
- **Goal:** After reordering, mask the assistant message (now in position 1) and keep the user message (now in position 2) for loss calculation.
- A single-turn conversation consists of (optionally) one system message, one user message, and one assistant message.
- Use `full_ids` (the complete tokenized conversation) and `prefix_lengths` (cumulative token counts per message) to identify message spans.
- After reordering, find the "user" role span (now in position 2) and copy those tokens to `labels`, while keeping all other positions as `IGNORE_TOKEN_ID = -100`.
- Set `attention_mask` to be a list of 1s for all valid tokens (before padding).

**Validation:**
- Run `bash scripts/check_exercises_3.sh` (or `python reverse_conversation_func.py`) to validate your implementation against golden answers.
- The script will report whether your implementation matches the expected output.

**Report (paste into the PDF):**
- A screenshot or console output showing that your implementation passed validation.
- Brief explanation of your implementation approach.
- **Conceptual Question**: After training a model with this masking approach (where assistant messages are masked and user messages contribute to loss), what would the model's behavior be? What might this training strategy be useful for? Provide at least one specific real-world application scenario.

### 4. (5 Points) Part 4 — Reverse Loss Masking (Multi-turn)

Extend your reverse loss masking implementation to support multi-turn conversations in `reverse_conversation_func.py`.

**Task Details:**
- Complete the TODO for multi-turn conversations (Exercise 4).
- **Key Mechanism**: The function **does NOT reorder messages** for multi-turn. Instead, it **automatically adds a system message** if not present:
  - Original: `[user: ...] [assistant: ...] [user: ...] [assistant: ...]`
  - After preprocessing: `[system: "You are a good state predictor."] [user: ...] [assistant: ...] [user: ...] [assistant: ...]`
- **Goal:** Mask all assistant and system tokens with `IGNORE_TOKEN_ID`, and only keep **user tokens** for loss calculation.
- Multi-turn conversations contain multiple (user, assistant) pairs, potentially preceded by system messages.
- The added system message provides context for the first user turn.
- Handle truncation: when conversations exceed `max_length`, `prefix_lengths` may overshoot `len(full_ids)`. Use `min()` to stay in bounds.
- Iterate through all messages and identify which spans correspond to user utterances.

**Validation:**
- Run `bash scripts/check_exercises_4.sh` (or `python reverse_conversation_func.py --multi-turn`) to validate.
- Ensure your implementation handles conversations with arbitrary numbers of turns.

**Report (paste into the PDF):**
- A screenshot showing that your multi-turn reverse masking implementation passed validation.
- Explanation of how you extended the single-turn reverse logic to handle multiple user turns in a conversation.
- **Conceptual Question**: After training with this masking strategy (where only user messages contribute to loss in a multi-turn conversation), what would the model learn? In what real-world scenarios might this training approach be useful? Provide at least 1 specific examples and explain the value of such a model.

### 5. (20 Points) Part 5 — Implementing Cross-Entropy Loss
>>>>>>> 846ec3382b0c09ac1adc60e65c032bec5399a431

Implement the token-level cross-entropy loss function in `loss_functions.py`.

**Task Details:**
- Complete the `cross_entropy_loss()` function in `loss_functions.py`.
- This function computes the language modeling loss from raw logits without using `torch.nn.CrossEntropyLoss`.
- You must implement the loss computation explicitly using log-softmax over the vocabulary dimension.

**Understanding `num_items_in_batch`:**

The `num_items_in_batch` parameter is crucial for proper loss normalization. Here's why:

In language modeling, we predict the next token for each position in the sequence. However, not all positions contribute to the loss:
1. **Padding tokens**: Sequences shorter than `max_length` are padded, and these positions should be ignored.
2. **Masked tokens**: In SFT, user prompts are masked with `IGNORE_TOKEN_ID = -100` and should not contribute to loss.
3. **Gradient accumulation**: When using gradient accumulation, we accumulate gradients over multiple micro-batches before performing an optimizer step. This allows us to simulate larger batch sizes on limited GPU memory.

**Example:**
```python
# Batch with 2 sequences, each length 4, vocab_size=1000
logits = torch.randn(2, 4, 1000)  # [batch, seq_len, vocab]
labels = torch.tensor([
    [5, 10, -100, 2],      # Sequence 1: position 2 is masked
    [3, -100, -100, -100]  # Sequence 2: positions 1-3 are masked/padding
])
```

The loss should be computed as: `total_loss / num_items_in_batch` to get the mean loss per valid token.

**Understanding Gradient Accumulation:**

In our training setup, we use gradient accumulation to simulate large batch sizes. Here's how it works:

- **Micro-batch**: The actual batch size per GPU (e.g., `BSZPERDEV=1`)
- **Gradient accumulation steps**: Number of micro-batches to accumulate before updating weights (e.g., 128 steps)
- **Effective batch size**: `micro_batch_size × gradient_accumulation_steps × num_gpus`

For example, with `BSZPERDEV=1`, `gradient_accumulation_steps=128`, and `1 GPU`:
- Effective batch size = 1 × 128 × 1 = 128

**Why use gradient accumulation?**
- Limited GPU memory: We can't fit a batch of 128 sequences in memory at once
- Stable training: Larger effective batch sizes provide more stable gradient estimates
- Solution: Process 128 micro-batches of size 1, accumulate their gradients, then update weights

**Important Note about `num_items_in_batch` and Gradient Accumulation:**

The **HuggingFace Trainer automatically handles gradient accumulation scaling** for you. Specifically:

- The `num_items_in_batch` parameter passed to your loss function **already includes the gradient accumulation factor**
- This means `num_items_in_batch` accounts for both the valid tokens in the current micro-batch AND the gradient accumulation steps
- The Trainer uses this to properly scale the loss so that gradients are correctly normalized across accumulation steps

**For the loss_functions_checker.py:**
- The checker validates your loss computation logic only
- It does NOT test gradient accumulation scenarios
- Focus on correctly computing the loss for valid (non-masked, non-padding) tokens
- The `num_items_in_batch` in the test cases is simply the count of valid tokens in that test batch

**Implementation Notes:**
- Do NOT use `torch.nn.CrossEntropyLoss` or `F.cross_entropy` directly.
- The function should return a scalar loss tensor.
- Make sure to handle the case where all tokens are masked (avoid division by zero).

**Validation:**
Run the loss function checker to validate your implementation:
```bash
python loss_functions_checker.py
```

The checker will test your implementation on several test cases with known reference values.

**Report (paste into the PDF):**
- Screenshot showing that all test cases passed.
- Brief explanation of how you computed the loss (formula and key steps).
- Explanation of what `num_items_in_batch` represents and why it's necessary.

<<<<<<< HEAD
### 4. (20 Points) Part 4 — Supervised Fine-Tuning
=======
### 6. (25 Points) Part 6 — Supervised Fine-Tuning
>>>>>>> 846ec3382b0c09ac1adc60e65c032bec5399a431

Perform supervised fine-tuning on the SmolLM2-135M model using the provided training script.

**Task Details:**
- Use the script `scripts/sft.sh` to launch training.
- The script uses DeepSpeed for distributed training with ZeRO Stage 2 optimization.
- Training data: smol-smoltalk-6k dataset (6000 conversational examples).
- Default settings: 3 epochs, learning rate 2e-5, batch size 128 (with gradient accumulation), max sequence length 2048.
- Training metrics will be logged to Weights & Biases (W&B) if configured; otherwise, they print to console.

**Running Training:**
```bash
cd /path/to/LLM-HW-2
bash scripts/sft.sh
```

**Key Training Arguments** (editable in `scripts/sft.sh`):
- `--num_train_epochs`: Number of epochs (default: 3)
- `--learning_rate`: Learning rate (default: 1e-5)
- `--warmup_ratio`: Warmup proportion (default: 0.1)
- `--lr_scheduler_type`: LR scheduler ("cosine", "linear", etc.)
- `--model_max_length`: Maximum sequence length (default: 2048)
- `TOTALBSZ`: Total effective batch size (default: 128)
- `BSZPERDEV`: Batch size per GPU (default: 1)

**Report (paste into the PDF):**
- Training configuration summary (GPU type, batch size, learning rate, epochs, total steps).
- Training loss curve (screenshot from W&B or console logs).
- Final checkpoint path.
- **Answer the question:** What is the role of `tokenizer.apply_chat_template()` in the training pipeline? How does it format conversations?

<<<<<<< HEAD
### 5. (35 Points) Part 5 — Instruction-Following Evaluation & Hyperparameter Tuning
=======
### 7. (35 Points) Part 7 — Instruction-Following Evaluation & Hyperparameter Tuning
>>>>>>> 846ec3382b0c09ac1adc60e65c032bec5399a431

Evaluate both the base model and your fine-tuned model using the IFEval benchmark, which tests instruction-following capabilities.

**Task Details:**
- IFEval contains prompts with specific formatting or content requirements (e.g., "Write at least 300 words", "Include exactly 3 paragraphs", "End with the word 'love'").
- Run evaluation on both:
  1. Base model: `SmolLM2-135M`
  2. Your fine-tuned model: checkpoint from Part 4
- Use the script `ifeval/run.sh`.

**Baseline Performance:**
The baseline strict accuracy for this task is approximately **22%**.

**Scoring Rubric:**
- **25-35 points**: Strict accuracy significantly higher than 22%, we will judge depending on the overall performance of all assignments.
- **25 points**: Strict accuracy > 22%
- **15 points**: Strict accuracy = 20-22%
- **5 points**: Strict accuracy = 15-20%
- **0 points**: Strict accuracy < 15%

**Running Evaluation:**
```bash
cd ifeval
# Evaluate base model
bash run.sh SmolLM2-135M results/SmolLM2-135M

# Evaluate fine-tuned model (replace path with your checkpoint)
bash run.sh /path/to/your/checkpoint results/SmolLM2-135M-SFT
```

**Hyperparameter Tuning:**

If your initial model does not exceed the baseline, you should tune hyperparameters. Consider adjusting:

1. **Learning Rate**: 
   - Higher LR: Faster learning but may be unstable
   - Lower LR: More stable but may need more epochs

2. **Number of Epochs**: 
   - Too few: Underfitting
   - Too many: Overfitting

3. **Warmup Ratio**: 
   - Controls the proportion of steps with learning rate warmup

4. **Learning Rate Schedule**

5. **Batch Size**: Try different total batch sizes 
   - Larger batch = more stable gradients but slower iteration

6. **Sequence Length**: 
   - Longer sequences = more context but more memory

**Report (paste into the PDF):**
- Comparison table showing IFEval metrics before and after SFT:
<<<<<<< HEAD
  - Prompt-level strict accuracy
  - Prompt-level loose accuracy
  - Instruction-level strict accuracy
  - Instruction-level loose accuracy
=======
  - strict accuracy
  - loose accuracy
>>>>>>> 846ec3382b0c09ac1adc60e65c032bec5399a431
- Final strict accuracy of your best model (must be > 22%)
- Hyperparameter tuning summary:
  - Table of different configurations you tried
  - Results (training loss, IFEval scores) for each configuration
  - Analysis of which hyperparameters had the most impact
- Example outputs: 2-3 interesting cases showing how behavior changed after SFT
- Analysis:
  - Which instruction-following capabilities improved?
  - Are there instruction types where the model still struggles?
  - Why do you think certain hyperparameters helped?

## Background

### Why Loss Masking?

In conversational data, each example contains both:
1. **User input (query)**: What the user asks
2. **Assistant output (response)**: How the model should respond

During SFT, we only want to optimize the model's ability to generate the assistant's response, not to predict the user's query. Therefore, we mask the loss on user tokens by setting their labels to `IGNORE_TOKEN_ID = -100`, which PyTorch's loss functions ignore by default.

**Example:**
```
User: What is the capital of France?
Assistant: The capital of France is Paris.
```

We compute loss only on tokens in "The capital of France is Paris." while masking "What is the capital of France?".

### Cross-Entropy Loss for Language Modeling

The cross-entropy loss measures how well the model's predicted probability distribution matches the true distribution (one-hot encoded target).

For a single token prediction:
```
loss = -log(P(correct_token))
```

Where `P(correct_token)` is the model's predicted probability for the correct next token.

For a batch with multiple tokens:
```
total_loss = sum(-log(P(correct_token_i)) for all valid tokens i)
average_loss = total_loss / num_valid_tokens
```

This is why `num_items_in_batch` (number of valid tokens) is crucial for computing the mean loss.

### Model and Dataset

- **Model**: [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) — A compact 135M parameter language model
- **Dataset**: [smol-smoltalk-6k](https://huggingface.co/datasets/PeterV09/smol-smoltalk-6k) — A 6000-sample subset of the original [smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) conversational dataset

### Chat Templates

Modern tokenizers use chat templates to format conversations consistently. The `apply_chat_template()` method converts a list of message dictionaries into a properly formatted token sequence with special tokens (e.g., `<|im_start|>`, `<|im_end|>` for chat protocols).

## Setup

### Installation

Run the setup script to install all dependencies and download resources:
```bash
bash setup.sh
```

The script will:
- Install PyTorch 2.8.0 with CUDA 12.6 support
- Install Transformers 4.57.1, vLLM 0.10.2, DeepSpeed
- Download NLTK punkt tokenizer data
- Download SmolLM2-135M model to `SmolLM2-135M/` directory
- Download smol-smoltalk-6k dataset to `smol-smoltalk-6k.json`

### GPU Requirements

- Recommended: At least 1 GPU with 16GB+ VRAM (e.g., NVIDIA 2080 Ti, T4, V100, A100)
- CPU-only training is possible but will be extremely slow
- Google Colab free tier with GPU is sufficient for this homework

## Important Notes

### Data Format

The training data should be in JSON or JSONL format with a `conversations` field:

```json
{
  "conversations": [
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is..."}
  ]
}
```

### Preprocessing and Caching

- The training script automatically preprocesses and caches tokenized data as pickle files.
- Cached features are reused across runs for faster startup.
- Manual cache regeneration: delete the `*_processed.pickle` file.

### Memory Optimization

If you encounter CUDA out-of-memory errors:
1. Reduce `BSZPERDEV` (batch size per device) in `sft.sh`
2. Reduce `--model_max_length` (e.g., from 2048 to 1024)
3. Use DeepSpeed CPU offloading: `ds_configs/zero2_offload.json`

### Hyperparameter Tuning Tips

- **Start small**: Test with fewer epochs first to iterate quickly
- **One at a time**: Change one hyperparameter at a time to understand its effect
- **Monitor training loss**: If it's not decreasing, try higher LR or more epochs
- **Watch for overfitting**: If training loss decreases but eval performance worsens, you may be overfitting
- **Document everything**: Keep a log of all experiments for your report

## Submission

Submit a zip of the codebase (only the homework directory) and a PDF report to Canvas. Your PDF should include your full name, student ID, and your UST email.

<<<<<<< HEAD
### Part 1: Single-turn Loss Masking (20 points)
- Screenshot showing validation passed
- Brief explanation of your implementation logic

### Part 2: Multi-turn Loss Masking (20 points)
- Screenshot showing validation passed
- Explanation of how you extended the single-turn logic

### Part 3: Cross-Entropy Loss Implementation (15 points)
=======
### Part 1: Single-turn Loss Masking (5 points)
- Screenshot showing validation passed
- Brief explanation of your implementation logic

### Part 2: Multi-turn Loss Masking (5 points)
- Screenshot showing validation passed
- Explanation of how you extended the single-turn logic

### Part 3: Reverse Loss Masking - Single-turn (5 points)
- Screenshot showing validation passed
- Brief explanation of your implementation approach
- **Conceptual Question**: After training a model with this masking approach (where assistant messages are masked and user messages contribute to loss), what would the model's behavior be? What might this training strategy be useful for? Provide at least one specific real-world application scenario.

### Part 4: Reverse Loss Masking - Multi-turn (5 points)
- Screenshot showing validation passed
- Explanation of how you extended the single-turn reverse logic to handle multiple user turns in a conversation
- **Conceptual Question**: After training with this masking strategy (where only user messages contribute to loss in a multi-turn conversation), what would the model learn? In what real-world scenarios might this training approach be useful? Provide at least 2 specific examples and explain the value of such a model.

### Part 5: Cross-Entropy Loss Implementation (20 points)
>>>>>>> 846ec3382b0c09ac1adc60e65c032bec5399a431
- Screenshot showing all test cases passed
- Explanation of your loss computation (formula and steps)
- Explanation of what `num_items_in_batch` represents and why it's necessary

<<<<<<< HEAD
### Part 4: Supervised Fine-Tuning (25 points)
=======
### Part 6: Supervised Fine-Tuning (25 points)
>>>>>>> 846ec3382b0c09ac1adc60e65c032bec5399a431
- Training configuration summary
- Training loss curve (screenshot)
- **Answer the question**: What is the role of `tokenizer.apply_chat_template()` in the SFT pipeline? How does it format conversations?

<<<<<<< HEAD
### Part 5: IFEval Evaluation & Tuning (15 points)
=======
### Part 7: IFEval Evaluation & Tuning (35 points)
>>>>>>> 846ec3382b0c09ac1adc60e65c032bec5399a431
- Comparison table (before/after SFT metrics)
- Final strict accuracy (must be > 22%)
- Hyperparameter tuning summary with results table
- Example outputs (2-3 cases)
- Analysis of improvements and hyperparameter effects

### Code Submission
- Include your implemented `conversation_func.py`
<<<<<<< HEAD
=======
- Include your implemented `reverse_conversation_func.py`
>>>>>>> 846ec3382b0c09ac1adc60e65c032bec5399a431
- Include your implemented `loss_functions.py`
- Include any modifications to `scripts/sft.sh`
- Do NOT include model checkpoints or cache files

## Quick Reference

```bash
# Setup
bash setup.sh

<<<<<<< HEAD
# Validate loss masking
bash scripts/check_exercises_1.sh  # Single-turn
bash scripts/check_exercises_2.sh  # Multi-turn

# Validate loss function
python loss_functions_checker.py

# Training
bash scripts/sft.sh

# Evaluation
=======
# Validate loss masking (Parts 1 & 2)
bash scripts/check_exercises_1.sh  # Part 1: Single-turn
bash scripts/check_exercises_2.sh  # Part 2: Multi-turn

# Validate reverse loss masking (Parts 3 & 4)
bash scripts/check_exercises_3.sh  # Part 3: Single-turn reverse
bash scripts/check_exercises_4.sh  # Part 4: Multi-turn reverse

# Validate loss function (Part 5)
python loss_functions_checker.py

# Training (Part 6)
bash scripts/sft.sh

# Evaluation (Part 7)
>>>>>>> 846ec3382b0c09ac1adc60e65c032bec5399a431
cd ifeval
bash run.sh SmolLM2-135M results/base
bash run.sh /path/to/checkpoint results/finetuned
```

## Repo Structure

```
.
<<<<<<< HEAD
├── conversation_func.py          # TODO: implement loss masking logic
├── loss_functions.py             # TODO: implement cross-entropy loss
├── loss_functions_checker.py     # Validation script for loss function
├── train_hw_parallel.py          # Main training script with DeepSpeed
├── utils.py                      # Helper functions
├── setup.sh                      # Environment setup
├── requirements.txt              # Dependencies
├── exercise_samples.json         # Test cases for loss masking
├── exercise_solutions.json       # Golden answers for loss masking
├── reference_answers.pkl         # Reference answers for loss function
├── scripts/
│   ├── sft.sh                   # Training launch script
│   ├── check_exercises_1.sh     # Single-turn validation
│   └── check_exercises_2.sh     # Multi-turn validation
├── ds_configs/                   # DeepSpeed configurations
├── ifeval/                       # Instruction-following evaluation
│   ├── run.sh                   # Evaluation script
│   └── run_ifeval.py            # Main evaluation logic
├── SmolLM2-135M/                # Base model (downloaded)
└── smol-smoltalk-6k.json        # Training dataset (downloaded)
=======
├── conversation_func.py              # TODO: implement loss masking logic (Parts 1 & 2)
├── reverse_conversation_func.py      # TODO: implement reverse loss masking (Parts 3 & 4)
├── generate_reverse_solutions.py     # Script to generate golden answers for reverse exercises
├── loss_functions.py                 # TODO: implement cross-entropy loss (Part 5)
├── loss_functions_checker.py         # Validation script for loss function
├── train_hw_parallel.py              # Main training script with DeepSpeed (Part 6)
├── utils.py                          # Helper functions
├── setup.sh                          # Environment setup
├── requirements.txt                  # Dependencies
├── exercise_samples.json             # Test cases for loss masking
├── exercise_solutions.json           # Golden answers for normal loss masking
├── reverse_exercise_solutions.json   # Golden answers for reverse loss masking
├── reference_answers.pkl             # Reference answers for loss function
├── scripts/
│   ├── sft.sh                       # Training launch script
│   ├── check_exercises_1.sh         # Part 1: Single-turn validation
│   ├── check_exercises_2.sh         # Part 2: Multi-turn validation
│   ├── check_exercises_3.sh         # Part 3: Single-turn reverse validation
│   └── check_exercises_4.sh         # Part 4: Multi-turn reverse validation
├── ds_configs/                       # DeepSpeed configurations
├── ifeval/                           # Instruction-following evaluation (Part 7)
│   ├── run.sh                       # Evaluation script
│   └── run_ifeval.py                # Main evaluation logic
├── SmolLM2-135M/                    # Base model (downloaded)
└── smol-smoltalk-6k.json            # Training dataset (downloaded)
>>>>>>> 846ec3382b0c09ac1adc60e65c032bec5399a431
```

## Troubleshooting

**1. "NotImplementedError: Exercise 1/2"**
- Complete the TODOs in `conversation_func.py`

<<<<<<< HEAD
**2. "NotImplementedError: Implement token-level cross-entropy"**
- Complete the `cross_entropy_loss()` function in `loss_functions.py`

**3. "CUDA out of memory"**
- Reduce `BSZPERDEV` or `--model_max_length` in `sft.sh`
- Use `ds_configs/zero2_offload.json` for CPU offloading

**4. "Validation failed"**
- Check that you're correctly identifying assistant tokens using `prefix_lengths`
- Ensure `IGNORE_TOKEN_ID` is used for masked positions

**5. Training loss is not decreasing**
=======
**2. "NotImplementedError: Exercise 3/4"**
- Complete the TODOs in `reverse_conversation_func.py`

**3. "NotImplementedError: Implement token-level cross-entropy"**
- Complete the `cross_entropy_loss()` function in `loss_functions.py`

**4. "CUDA out of memory"**
- Reduce `BSZPERDEV` or `--model_max_length` in `sft.sh`
- Use `ds_configs/zero2_offload.json` for CPU offloading

**5. "Validation failed"**
- Check that you're correctly identifying assistant/user tokens using `prefix_lengths`
- Ensure `IGNORE_TOKEN_ID` is used for masked positions
- For reverse masking (Parts 3 & 4), ensure you're masking assistant turns instead of user turns

**6. Training loss is not decreasing**
>>>>>>> 846ec3382b0c09ac1adc60e65c032bec5399a431
- Increase learning rate (try 2e-5 or 5e-5)
- Increase number of epochs
- Check that loss function is implemented correctly

<<<<<<< HEAD
**6. IFEval score is below 22%**
=======
**7. IFEval score is below 22%**
>>>>>>> 846ec3382b0c09ac1adc60e65c032bec5399a431
- Try different learning rates (1e-5 to 5e-5)
- Try more epochs (4-5 epochs)
- Try different batch sizes
- Check your loss masking implementation
- Ensure training loss is decreasing properly

<<<<<<< HEAD
**7. Training is very slow**
=======
**8. Training is very slow**
>>>>>>> 846ec3382b0c09ac1adc60e65c032bec5399a431
- Increase `BSZPERDEV` if GPU memory allows
- Reduce `--model_max_length` to process sequences faster
- Enable Flash Attention in `sft.sh` if supported
- Use multiple GPUs if available

<<<<<<< HEAD
**8. Loss function checker fails**
=======
**9. Loss function checker fails**
>>>>>>> 846ec3382b0c09ac1adc60e65c032bec5399a431
- Make sure you're using `F.log_softmax()` for numerical stability
- Check that you're handling the causal shift correctly (logits[i] predicts labels[i+1])
- Verify you're masking positions where `labels == -100`
- Ensure you're dividing by `num_items_in_batch` for normalization

For more help:
- Post questions on Canvas
- Review code comments in source files
- Use AI tools to help understand concepts
