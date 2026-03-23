# Agent Instructions

**Language Requirements:**
- All code, scripts, variables, and technical documentation must be written in English. 
- All direct communication with the user must be in Danish. 

You're working inside the **WAT framework** (Workflows, Agents, Tools). 
This architecture separates concerns so that probabilistic AI handles reasoning while deterministic code handles execution. 
That separation is what makes this system reliable. 

## The WAT Architecture

**Layer 1: Workflows (The Instructions)**
- Markdown SOPs stored in `workflows/`
- Each workflow defines the objective, required inputs, which tools to use, expected outputs, and how to handle edge cases
- Written in plain language, the same way you'd brief someone on your team

**Layer 2: Agents (The Decision-Maker)**
- This is your role. You're responsible for intelligent coordination. 
- Read the relevant workflow, run tools in the correct sequence, handle failures gracefully, and ask clarifying questions when needed
- You connect intent to execution without trying to do everything yourself
- Example: Read `workflows/scrape_website.md`, figure out the required inputs, then execute `tools/scrape_single_site.py` 

**Layer 3: Tools (The Execution)**
- Python scripts in `tools/` that do the actual work
- API calls, data transformations, file operations, database queries
- Credentials and API keys are stored in `.env`
- These scripts are consistent, testable, and fast

**Why this matters:** When AI tries to handle every step directly, accuracy drops fast. If each step is 90% accurate, you're down to 59% success after just five steps. By offloading execution to deterministic scripts, you stay focused on orchestration and decision-making where you excel. 

## How to Operate

**1. Look for existing tools first** 
Before building anything new, check `tools/` based on what your workflow requires. Only create new scripts when nothing exists for that task. 

**2. Learn and adapt when things fail** 
When you hit an error:
- Read the full error message and trace
- Fix the script and retest
- Document what you learned in the workflow 

**3. Keep workflows current** 
Workflows should evolve as you learn. However, don't create or overwrite workflows without asking unless I explicitly tell you to. These are your instructions and need to be preserved and refined, not tossed after one use. 

## The Self-Improvement Loop

Every failure is a chance to make the system stronger. Identify what broke, fix the tool, verify the fix works, update the workflow with the new approach, and move on with a more robust system. 

## File Structure

**What goes where:**
- **Deliverables**: Final outputs go to cloud services where I can access them directly
- **Intermediates**: Temporary processing files that can be regenerated

**Directory layout:**
- `.tmp/`: Temporary files. Regenerated as needed. Everything in `.tmp/` is disposable. 
- `tools/`: Python scripts for deterministic execution
- `workflows/`: Markdown SOPs defining what to do and how
- `.env`: API keys and environment variables

Local files are just for processing. Anything I need to see or use lives in cloud services. 

## Bottom Line

You sit between what I want (workflows) and what actually gets done (tools). Your job is to read instructions, make smart decisions, call the right tools, recover from errors, and keep improving the system as you go. Stay pragmatic. Stay reliable. Keep learning. 

---

## CURRENT MISSION: Shopping Center Anomaly Detection

**Objective:** Build a data pipeline and a Streamlit web application to detect anomalies (like small, constant leaks or running toilets) in electricity and water usage.

**Data Context:**
- We have 3 years of hourly data spread across 23 separate CSV files.
- The data is multivariable: Electricity consumption, Water consumption, and Outside Temperature.

**Execution Rules for this Mission:**
1. **Data Wrangling First:** Before building any machine learning models, write deterministic tools to merge the 23 separate files based on their timestamps into a single, structured dataset.
2. **Train/Test Split:** The model must NOT be trained on the entire dataset. Use Year 1 and Year 2 data to establish baseloads, seasonality, and normal usage patterns. Year 3 data must be kept separate for testing/validation.
3. **Machine Learning Approach:** Use algorithms suited for multivariable time-series anomaly detection (e.g., Isolation Forest). Rely on the outside temperature data to explain seasonal spikes (like HVAC usage) to avoid false positives.
4. **UI Deliverable:** Develop a Streamlit application where the user can upload files, run the trained model on new test data, and view interactive charts where detected anomalies are highlighted in red.
