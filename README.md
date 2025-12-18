# Agentic Research Assistant

An agentic research assistant that interprets high-level research goals, generates execution plans using a large language model (LLM), and carries out mode-aware research workflows using both classical machine learning and LLM-based tools.  
The system supports multiple research goals and maintains structured memory across multi-step executions.

---

## Project Overview

This project implements a lightweight **agentic architecture** consisting of:

- A goal-driven **planner** powered by an LLM
- A deterministic **executor** that maps planned steps to concrete tools
- A shared **memory** that stores intermediate and final results
- Mode-aware execution paths for different types of research tasks

The assistant currently supports two research workflows:
1. **Text Classification** using classical machine learning
2. **Research Paper Summarization** from PDF documents using an LLM

The focus of the project is on **agent behavior, planning–execution separation, and robust workflow control**, rather than model training at scale.

---

## Supported Research Goals

### 1. Text Classification
- Loads a real-world text dataset
- Preprocesses text into features
- Trains a Naive Bayes classifier
- Evaluates model performance
- Stores observations in memory

### 2. Research Paper Summarization
- Loads local research papers in PDF format
- Extracts text from PDFs
- Performs token-safe, chunked summarization using an LLM
- Stores summaries and research observations in memory

The research mode is automatically inferred from the user’s goal.

---

## System Architecture

```
User Goal
   ↓
Research Mode Inference
   ↓
LLM Planner (Task Decomposition)
   ↓
Executor (Mode-Aware, Idempotent)
   ↓
Tools (ML / PDF / LLM)
   ↓
Shared Memory
```

### Key Design Principles
- **Planner–Executor Separation**: Planning is handled by an LLM, execution is deterministic.
- **Mode Awareness**: Only valid actions are executed for a given research mode.
- **Idempotent Execution**: Resources such as datasets and models are loaded or trained only once.
- **Explicit Failure Handling**: The system safely skips or fails tasks when prerequisites are missing.

---

## How to Run

### 1. Set up the environment

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

---

### 2. Run the assistant (interactive mode)

```bash
python src/main.py
```

You will be prompted to choose a research goal.

---

### 3. Run the assistant with a direct goal

```bash
python src/main.py --goal "Build a machine learning text classification model"
```

or

```bash
python src/main.py --goal "Summarize recent research papers on transformers"
```

---

## Example Execution Flow

1. User provides a research goal
2. Research mode is inferred (classification or summarization)
3. Planner generates a multi-step plan
4. Executor executes each step using appropriate tools
5. Intermediate results are stored in memory
6. Final outputs (metrics or summaries) are displayed

The generated plan and execution results are printed to the console for transparency.

---

## Tools and Technologies Used

- **Python**
- **scikit-learn** (classical ML)
- **PyMuPDF** (PDF text extraction)
- **LLM API** (planning and summarization)
- **argparse** (CLI interface)

---

## Limitations

- The assistant does not perform web search or live paper retrieval
- PDFs must be provided locally
- The system is designed for research prototyping, not production deployment
- No web or GUI interface is included

These constraints are intentional to keep the system focused and explainable.

---

## Motivation

The goal of this project is to demonstrate:
- Agentic workflow design
- Practical integration of LLMs with deterministic systems
- Robust handling of real-world constraints such as token limits and missing data
- Clear separation of reasoning, execution, and state


