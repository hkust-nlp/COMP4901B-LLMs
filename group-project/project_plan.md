## Project Plan for COMP4901B Group Project

### 1. Overall Phases

1. Environment & Setup  
2. Part I – Search-Augmented QA Agent  
3. Part II – Realistic Multi-Tool Agent  
4. Experiments & Evaluation  
5. Report & Submission Packaging  

Assume **2 teammates**: `Teammate A` and `Teammate B` (rename as needed).

---

## 2. Environment & Repo Setup (Day 1)

### 2.1 Project Setup

- **Step 1 – Clone & structure check**
  - Ensure repo matches the given structure:
    - `README.md`, `requirements.txt`, `data/nq_test_100.jsonl`, `src/`, `scripts/`, `results/`
  - Confirm `scripts/grade_with_em.py` and `scripts/grade_with_llm_judge.py` exist and **do not modify** them.

- **Step 2 – Create Python environment**
  - Install `uv` (or use your own tool, but follow project recommendation).
  - In project root (`group-project`):

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

- **Step 3 – Configure API keys**
  - Create something like `.env` or a `config.py` (not committed) that stores:
    - `SERPER_API_KEY`
    - `DEEPSEEK_API_KEY`
  - Decide a **standard way** to load keys (e.g., `os.environ`).
  - Ensure both teammates can run a tiny test script to:
    - Call DeepSeek (chat completion) once.
    - Call Serper (search) once.

- **Suggested ownership**
  - **Teammate A**: Lead on environment setup, DeepSeek test.
  - **Teammate B**: Lead on Serper search test, `.env`/config conventions.

---

## 3. Part I – Search-Augmented LLM Agent (70 points)

### 3.1 Implement Baseline Agent (No Search)

Goal: A basic QA model using **DeepSeek-chat** only, generating answers from question + context (no external tools).

- **Step 1 – Design `src/agent.py` structure**
  - Suggested classes/functions:
    - `class BaseAgent:`
      - `answer_question(question: str) -> str`
    - `class NoSearchAgent(BaseAgent):`
      - Uses DeepSeek-chat API directly.
  - Decide how to:
    - Set model name (`deepseek-chat`).
    - Set base URL and API key (read from env).
    - Set prompt template for QA (instruction + question).

- **Step 2 – Implement DeepSeek wrapper**
  - A function like:
    - `call_deepseek(messages: List[dict], tools: Optional[List[dict]] = None)`.
  - Handles basic errors (even simple is fine).

- **Step 3 – Implement prediction script for no-search**
  - Script or function:
    - Loads `data/nq_test_100.jsonl`.
    - For each example:
      - Get `question`, `id`, `answers` (if provided).
    - Call `NoSearchAgent.answer_question`.
    - Save outputs to `results/predictions_nosearch.jsonl` with **exact same format** as `results/prediction_example.jsonl`.

- **Division**
  - **Teammate A**: Design & implement `NoSearchAgent`, DeepSeek wrapper, JSONL prediction loop.
  - **Teammate B**: Review prompts, test quality on a few examples, adjust temperature/max_tokens.

---

### 3.2 Implement Search-Augmented Agent (Core of Part I)

Goal: An **agent loop** with Serper search and DeepSeek reasoning.

- **Step 1 – Define tools**
  - `search_tool(query: str) -> List[documents]`
    - Call Serper API.
    - Return titles, URLs, snippets, maybe top `k=5`.
  - Optional: in-code tool definitions for DeepSeek tool-calling (name, description, parameters).

- **Step 2 – Design agent loop**
  - Define:
    - Maximum search steps (e.g., 3–5).
    - Termination condition (e.g., model chooses an `answer` action or hits step limit).
  - Example loop:
    1. Model sees question + current notes + available tools.
    2. Model decides:
       - Call `search` with some query, or
       - Output final `answer`.
    3. If `search`, you call Serper, append results to context, and ask the model again.
    4. Stop when `answer` tool is called or step limit reached.

- **Step 3 – Implementation in `agent.py`**
  - `class SearchAgent(BaseAgent):`
    - `run_trajectory(question: str) -> dict`  
      - Returns full trajectory (steps, actions, observations).
    - `answer_question(question: str) -> str`  
      - Internally uses `run_trajectory` and returns final answer.
  - Make design **simple but clear** so you can screenshot it for the report.

- **Step 4 – Save predictions and trajectories**
  - For each question in `nq_test_100.jsonl`:
    - Run the search agent.
    - Save:
      - Answer to `results/predictions_search.jsonl` in example format.
      - Full trajectory to `results/agent_trajectories.jsonl` following exactly `results/trajectories_example.jsonl`.

- **Division**
  - **Teammate B (Lead)**:
    - Implement `search_tool` (Serper wrapper).
    - Implement `SearchAgent` agent loop and trajectory structure.
  - **Teammate A**:
    - Integrate DeepSeek tool-calling (if you use function calling).
    - Help design prompts so the model knows when to search and when to answer.
    - Implement saving of trajectories JSONL.

---

### 3.3 Evaluation & Tuning

Goal: Reach required thresholds and understand behavior.

- **Step 1 – Run EM script**
  - For **both**:
    - `results/predictions_nosearch.jsonl`
    - `results/predictions_search.jsonl`

```bash
PYTHONPATH=. python scripts/grade_with_em.py \
    --input results/predictions_nosearch.jsonl \
    --output results/grading_nosearch_em.json

PYTHONPATH=. python scripts/grade_with_em.py \
    --input results/predictions_search.jsonl \
    --output results/grading_search_em.json
```

- **Step 2 – Run LLM judge**
  - For both prediction files:

```bash
PYTHONPATH=. python scripts/grade_with_llm_judge.py \
    --input results/predictions_nosearch.jsonl \
    --model deepseek-chat \
    --base_url https://api.deepseek.com/v1 \
    --api_key YOUR_DEEPSEEK_KEY \
    --output results/grading_nosearch_llm_judge.json

PYTHONPATH=. python scripts/grade_with_llm_judge.py \
    --input results/predictions_search.jsonl \
    --model deepseek-chat \
    --base_url https://api.deepseek.com/v1 \
    --api_key YOUR_DEEPSEEK_KEY \
    --output results/grading_search_llm_judge.json
```

- **Step 3 – Tune hyperparameters**
  - Adjust:
    - Max tokens, temperature.
    - Number of search results `k`.
    - Max search steps.
  - Re-run evaluation until:
    - No-search: EM > 36%, LLM judge > 65%.
    - With search: LLM judge at least 5 points higher than no-search.

- **Step 4 – Select example trajectories for report**
  - Find **2 trajectories** where search clearly helps.
  - Save their IDs and steps for screenshots and explanation.

- **Division**
  - **Teammate A**:
    - Lead evaluation scripts & log organization.
    - Summarize EM/LLM judge numbers for baseline (no-search).
  - **Teammate B**:
    - Lead evaluation and tuning for search agent.
    - Identify and annotate the 2 good trajectories.

---

### 3.4 (Optional) Bonus: Browsing Tool

If you have time:

- **Step 1 – Define `browse` tool**
  - Given a URL, fetch full HTML and extract main text (e.g., using `requests` + `BeautifulSoup`).

- **Step 2 – Integrate into agent**
  - Allow agent to:
    - Use `search` → get URLs.
    - Use `browse` on selected URLs to get full content.

- **Step 3 – Evaluate again**
  - Produce new predictions + trajectories.
  - Compare EM + LLM judge to search-only.

- **Division**
  - Mainly **Teammate B**, with **Teammate A** assisting on evaluation and report writing for bonus.

---

## 4. Part II – Realistic Agent with ≥3 Tools (30 points)

### 4.1 Decide Scenario and Tools

- **Step 1 – Brainstorm realistic workflows**
  - Example themes:
    - **Personal productivity assistant**:
      - Tools: Google Calendar API, Google Tasks / Notion, Gmail.
    - **Developer assistant**:
      - Tools: GitHub Issues/PRs, Slack, Google Calendar/Drive.
    - **Travel planning assistant**:
      - Tools: Google Maps, Calendar, Notes/Sheets.

- **Step 2 – Fix on at least 3 tools**
  - Example concrete choice:
    - `calendar_tool` – Create events, list schedule.
    - `tasks_tool` – Manage todo items in Notion or Google Tasks.
    - `maps_tool` – Find locations / travel time via Google Maps.

- **Step 3 – Divide integration work**
  - Each tool will need:
    - API credentials and setup.
    - A Python wrapper (`*_tool`).
    - A schema for DeepSeek tool-calling.

- **Division**
  - **Teammate A**: Integrate Tool 1 (e.g., Calendar) and Tool 2 (e.g., Tasks).
  - **Teammate B**: Integrate Tool 3 (e.g., Maps) and build agent loop.

---

### 4.2 Implement Realistic Agent

- **Step 1 – Common tool interface**
  - Create something like:
    - `tools.py` with functions/classes per tool.
    - Clear docstrings: what each tool does, parameters, return format.

- **Step 2 – Multi-tool agent design**
  - `class RealWorldAgent:`
    - Uses DeepSeek with tool-calling.
    - Maintains a conversation history and state.
    - At each step:
      - Model decides which tool to call (or respond to user).
      - Call the tool in Python, append the result to messages, continue.

- **Step 3 – Define 3 tasks (for the report)**
  - Each task:
    - Is a **realistic** scenario.
    - Uses **≥3 tools**.
    - Has **≥5 steps**.
  - Examples:
    - Task 1: Plan a study week:
      - Look up class schedule (Calendar).
      - Add study sessions (Calendar).
      - Create todos (Tasks).
    - Task 2: Plan a meeting:
      - Check free slots (Calendar).
      - Find a cafe nearby (Maps).
      - Create a task list (Tasks).
    - Task 3: Weekend outing:
      - Check weather (a 4th tool, optional).
      - Suggest locations (Maps).
      - Create event (Calendar) and packing list (Tasks).

- **Step 4 – Save trajectories**
  - Log full trajectories (calls, responses, tool outputs) for **all 3 tasks** for the report.
  - Could be JSONL or structured logs you later screenshot.

- **Division**
  - **Teammate B**:
    - Lead agent loop for Part II, tool-calling integration.
    - Ensure each trajectory uses ≥3 tools and ≥5 steps.
  - **Teammate A**:
    - Implement wrappers for chosen tools, handle auth.
    - Design and run the 3 demonstration tasks & save logs.

---

## 5. Report & Documentation

### 5.1 Report Structure (Match README Numbering)

1. **1. Search-Augmented LLM Agent**
   - Explain agent loop and termination condition.
   - Screenshot main implementation in `agent.py` (no-search and search versions).
   - Explain core components: DeepSeek wrapper, search tool, loop.

2. **2.1 Baseline (No Search) Results**
   - Describe:
     - EM score (>36%).
     - LLM judge score (>65%).
   - What you tried to reach those numbers (prompts, hyperparameters).
   - Screenshot of EM and LLM judge results.

3. **2.2 Search Agent Results**
   - Report EM + LLM judge scores.
   - Show that LLM judge accuracy is ≥5 points higher than baseline.
   - Screenshot EM and LLM judge outputs.
   - Reference where `agent_trajectories.jsonl` is saved.
   - Pick and describe 2 trajectories where search clearly helps; explain why.

4. **2.3 Analysis**
   - Do you get improved EM? Why/why not?
   - Do you get improved LLM judge? Why?
   - Use the 2 chosen trajectories as case studies.

5. **3. Bonus (if done)**
   - Describe `browsing` tool.
   - Screenshot key browsing code and evaluation results.
   - Discuss impact vs search-only.

6. **4. Part II – Realistic Multi-Tool Agent**
   - Describe:
     - The 3 (or more) tools and why you chose them.
     - What real workflows they support.
   - Screenshot main agent logic (tool-calling loop).
   - For each of the 3 tasks:
     - Show the full trajectory (at least 5 steps).
     - Explain if the agent succeeded; if not, what went wrong.

7. **5. Team Contributions**
   - Names, IDs.
   - Contribution percentages.
   - Brief bullet list of who did what.

- **Division**
  - **Teammate A**:
    - Draft sections 1, 2.1, 2.3, 5.
  - **Teammate B**:
    - Draft sections 2.2, 3 (if done), 4.
  - Both: Cross-review each other’s sections for consistency.

---

## 6. Timeline Suggestion (Before 2025-12-04 Deadline)

- **Day 1–2**
  - Env setup, API keys, DeepSeek + Serper smoke tests.
  - Design `agent.py` structure and basic wrappers.

- **Day 3–4**
  - Implement `NoSearchAgent` and `SearchAgent`.
  - Get first versions of `predictions_nosearch.jsonl` and `predictions_search.jsonl`.

- **Day 5–6**
  - Run EM + LLM judge.
  - Tune prompts/hyperparameters.
  - Save final predictions + trajectories.
  - Select example trajectories.

- **Day 7–8**
  - Decide Part II tools and workloads.
  - Implement tool wrappers and multi-tool agent.

- **Day 9–10**
  - Run 3 demonstration tasks and log trajectories.
  - If time: implement browsing bonus.

- **Day 11–12**
  - Write and polish report (both teammates).
  - Final sanity check:
    - Predictions file formats.
    - All required JSONL exist.
    - Scripts run without modification.

---

## 7. Final Work Division Summary

- **Teammate A (Model & Evaluation Lead)**
  - Environment + DeepSeek wrapper.
  - `NoSearchAgent` and baseline predictions script.
  - EM & LLM judge evaluation for baseline and search agent.
  - Help tune prompts/hyperparameters.
  - Part of tool implementation for Part II (e.g., Calendar, Tasks).
  - Report sections: 1, 2.1, 2.3, team contributions.

- **Teammate B (Agent & Tools Lead)**
  - Serper search wrapper and `SearchAgent` loop.
  - Trajectory logging for Part I.
  - Hyperparameter tuning for search agent and picking good trajectories.
  - Design and implement Part II multi-tool agent (core loop).
  - Integrate at least one major tool (e.g., Maps) and orchestrate 3 tasks.
  - Report sections: 2.2, 3 (optional), 4.


