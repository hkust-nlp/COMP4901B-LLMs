# Group Project: Search Agents and Beyond

**Full Score: 100 points**   
**Due Date: 2025-12-4 11:59 PM**

This group project aims to facilitate developing tool-using agents based on LLM APIs. The first part is to incorporate remote search functionalities (Google Search via Serper API) into LLMs. The second part is to incorporate other open-ended tools and applications.

## Project Overview

In this project, you will implement a **search-based QA system** where an LLM agent can:
1. **Reason** about complex questions
2. **Search** for relevant information using Google Search
3. **Iterate** by performing multiple search steps
4. **Synthesize** answers from retrieved documents

Unlike typical RAG (Retrieval-Augmented Generation) systems that perform a single retrieval step, your agent will learn to:
- Decide when to search for more information
- Formulate effective search queries
- Reason over multiple retrieved documents
- Determine when it has enough information to answer
  
> If you never heard about RAG before, that is fine and not required to know for this project, but it is recommended to simply ask ChatGPT what RAG is for you.


## General Rules
1. Each group should have 2-3 students, not more than 3. In case you feel difficult to find a teammate, you can choose to work alone. You will not be penalized by working alone, but you will not get bonus either, as we aim to bring students to work together on this project. **Please sign up to form the group on Canvas [here](https://canvas.ust.hk/courses/64649/groups).**
2. Different from other regular homework, this project is designed to be more open-ended, thus we don't provide detailed step-by-step codebases for you to only fill in some functions as in other homework. Instead, you need to implement the entire thing from scratch. This gives you more flexibility, and you don't need to stick to any environment setup, as long as the project goals are consistent.  
3. Following the first point, sometimes we deliberately do not provide detailed instructions on how to use some tools or APIs, instead we provide documentation references for you to grab yourself. Consider this as a necessary skill for LLM projects now and in the future. 
4. This project doesn't need GPUs, thus you are free to use any CPU-based machine to run your code, such as the Google Colab, the CSE cluster, your own laptop (mac or windows), etc. Particularly, if you are using a windows machine, you may need to make certain adjustments to the environment setup and commands.



## Project Structure

```
COMP4901B-project/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── data/
│   └── nq_test_100.jsonl          # NQ evaluation set 
├── src/
│   ├── agent.py                   # Agent implementation (YOU IMPLEMENT THIS)
│   ├── metrics.py                 # Evaluation metrics (provided)
│   └── ...                        # Other files you may need to implement
├── scripts/
│   ├── grade_with_em.py           # EM-based grading script
│   └── grade_with_llm_judge.py    # LLM judge grading script
└── results/                        # Evaluation results (generated)
```

## Submission Guidelines
-  Please indicate the names and IDs of both teammates in the PDF report. Please also indicate the contribution percentage of each teammate in the project at the beginning of the PDF report (e.g., 50% for each teammate). If the contribution of each is around 50%, each of you will get exactly the same score. However, if the contribution deviates far from 50%, we will give different scores.
-  Please sign up to form the group on Canvas [here](https://canvas.ust.hk/courses/64649/groups), then i think Canvas will automatically manage group submission

You are expected to submit two files:  
-   The zip file of the entire codebase (plz remove unnecessary outputs files before you zip). Name it as `[your-student-id]-[your-ust-username]-project.zip`  
- A **separate PDF report**, name it as `[your-student-id]-[your-ust-username]-report.pdf` (Please remember to submit this separately, you will get a penalty on the grading if you zip this PDF report together with the codebase)


## 🚀 Getting Started

### 1. Installation
> We recommend you to use uv to install the dependencies, which is a fast Python package installer and resolver to manage environment, compared to conda. However, you are free to use any other package manager you prefer, but below we only provide uv instructions. 

**Step 1: Install uv**

First, install `uv` - a fast Python package installer and resolver. Visit the official repository for installation instructions:

👉 **[uv Installation Guide](https://github.com/astral-sh/uv#installation)**

Quick install (macOS/Linux):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Step 2: Create Environment and Install Dependencies**

```bash
# Create a virtual environment with uv
uv venv

# Activate the environment
source .venv/bin/activate 

# Install dependencies
uv pip install -r requirements.txt
```

**Note**: After activating the environment with `source .venv/bin/activate`, you can run all Python commands normally (e.g., `python script.py`). The environment stays active in your current terminal session.

**⚠️ Important**: Make sure to activate the environment (`source .venv/bin/activate`) before running any of the commands below! You'll need to do this once per terminal session.

### 2. Set Up API Keys

For this project, you'll use remote search via the Serper API for Google Search and the latest DeepSeek-v3.2 model for LLMs. We have provided these keys in Canvas annoucements. 

**Important: Please keep these keys confidential and do not share them with anyone outside the class. Also, if use public repos to store your code, please make sure to remove the keys from the code before pushing to the repo.**


## Part I: Search-Augmented LLM Agents for Question Answering (70 points)

### Tasks:

1. (30 Points) Using Google Search via Serper API, implement a search-augmented LLM agent for question answering. Note that this is not just a single-time search to look for helpful information, your model needs to function in an agent loop as we have learned in class, and the model is expected to conduct multiple search steps to answer a question. What is the agent loop termination condition in your implementation?
    > Please screenshot your main implementation logic in the code to include in the PDF report, and explain the core components in the PDF report.
   
2. (40 points) Perform correct generation of the basic LLM (DeepSeek-v3.2 chat model) without search, and that with search, obtaining expected results on the Natural Question test set (we only include a subset for fast eval) in `data/nq_test_100.jsonl`. Please save your model predictions into `results/predictions_nosearch.jsonl` and `results/predictions_search.jsonl` respectively, following exactly the same format as `results/predictions_example.jsonl`  
   
   (1) (10 points) Evaluating the DeepSeek-v3.2 chat model (without search) with our provided evaluation script, the EM (Exact Match) score should be > 36%, the LLM judge accuracy should be > 65%. What did you try to get the desired scores? Any findings? 
    > Please screenshot your evaluation results to include in the PDF report.

   (2) (20 points) Evaluating your implemented LLM agent with search, save agent trajectories into `results/agent_trajectories.jsonl` following the exact format as `results/trajectories_example.jsonl`. Report the EM score and LLM judge accuracy. The LLM judge accuracy should be at least 5 absolute points higher than the results without search. 
   > Please screenshot your evaluation results to include in the PDF report.

   (3) (10 points) Do you get improved EM accuracy from search? Why or Why Not? Do you get improved LLM judge accuracy from search? Please showcase 2 agent trajectories and explain why doing search improves these cases.


3. (Bonus 10 points, this is optional) You may find that the default Google Search does not return full web page information, instead only returns title and a very short snippet. A potential way to improve this is to add another `browsing` tool besides the `search` tool, so that the agent can autonomously choose to browse full contents of the web page. Implement this advanced search agent with browsing functionalities, and report the EM score and LLM judge accuracy. Does it help over search-only agents? Why or Why not? 
    > Similar to the questions above, besides answering the questions, please screenshot your `browsing` implementation logic in the code to include in the PDF report, and explain the core components in the PDF report. Also screenshot the evaluation results to include in the PDF report.


#### Tips
* There are many hyperparameters to vary, such as how many search results to return, max generation tokens and max search steps of your agents.
* If you cannot obtain the desired EM score, please check your model predictions to understand the reason, and think of how you can improve the generation. 
* For using Serper API, you can refer to the [Serper Playground](https://serper.dev/playground), where you may need to register (for free) and check the `code` tab. For using DeepSeek API, please call it through the OpenAI package as you installed in previous steps, and refer to [here](https://api-docs.deepseek.com).
* [LLM tool calling through API](https://api-docs.deepseek.com/guides/function_calling)


#### Submission Report Requirement
1. Your report should follow the numbering exactly as the tasks, like 1, 2.1, 2.2, 3, etc. 

2. Answer all the questions asked in the PDF report. Besides, in each of the task above, we also made it clear what to include in the PDF report besides the questions. 

### Evaluation on Natural Questions
*Please do not change our provided evaluation scripts, they should be fine as long as your saved predictions follow our required formats*

**Exact Match (EM) Score**
```bash
PYTHONPATH=. python scripts/grade_with_em.py \
    --input [your-saved-jsonl-file-of-predictions] \
    --output grading_results_em.json
```
    
**LLM-as-Judge**
```bash
PYTHONPATH=. python scripts/grade_with_llm_judge.py \
    --input [your-saved-jsonl-file-of-predictions] \
    --model deepseek-chat \
    --base_url https://api.deepseek.com/v1 \
    --api_key YOUR_DEEPSEEK_KEY \
    --output grading_results_llm_judge.json
```

## Part II: Realistic Agent with More Flexible Tool Calling (30 points)

### Tasks:

1. (15 points) Implement a realistic agent with at least 3 tools. The tools should come from realistic applications from your daily life (rather than implementing some fake tools on your own). The applications can include Google calendar, Google maps, Notion, Slack, Github, Google sheet, etc. Your agent can support the previous google search tool as well, but that search tool does not count towards the 3-tool minimum requirements. **Please think of any realistic workflows that you would do manually and now try to build an agent to do it automatically!**
    > Please screenshot your main implementation logic in the code to include in the PDF report, and explain the core components in the PDF report. Also explain why you choose these tools, what kinda scenarios you are thinking.

2. (15 points) Demonstrate three realistic tasks that you would do manually in reality, and now using your implemented agent to do it automatically. Demonstrate the complete agent trajectories in your report and explain them. Did your agent complete these tasks correctly? If not, what went wrong? Each of these trajectories should be at least **5** steps long using at least 3 different tools. Also, the three tasks should be different, rather than just changing parameters or arguments of the same task (e.g., adding a new event to the calendar vs adding a new event to the calendar with a different time).

### Rubrics
We will grade dependending on the reality and complexity of your tools/tasks. Generally agent demonstratation that actually uses multiple advanced tools to address practical, complex tasks is preferred.

