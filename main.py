import os
import re
import sys
import io
import traceback
from typing import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# Load API keys from .env file
load_dotenv()

# 1. Define the State (The "Memory" of the AI)
class AgentState(TypedDict):
    task: str
    current_code: str
    error: str
    iterations: int

# Initialize the blazing-fast Groq model (Llama 3.1)
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)

# 2. Node: The Coder
def coder_node(state: AgentState):
    print(f"\n[Coder] Attempt {state['iterations'] + 1}...")
    
    if state["error"]:
        # We add a specific instruction to avoid the failing library
        prompt = f"""You are an expert Python engineer. 
Your last code failed with this error: {state['error']}

CRITICAL INSTRUCTION:
The environment does NOT have 'scipy' or external math libraries installed. 
You MUST rewrite the logic using only standard Python libraries (math, statistics, or raw loops).
Do NOT include 'import scipy' anywhere in your code.

Task: {state['task']}
Output ONLY the raw Python code inside ```python ``` blocks."""
    else:
        prompt = f"""You are an expert Python engineer. 
Write a Python script to: {state['task']}
Output ONLY the raw Python code inside ```python ``` blocks."""

    response = llm.invoke(prompt)
    
    code_match = re.search(r"```python\n(.*?)\n```", response.content, re.DOTALL)
    extracted_code = code_match.group(1) if code_match else response.content
        
    return {
        "current_code": extracted_code, 
        "iterations": state["iterations"] + 1
    }

# 3. Node: The Executor
def executor_node(state: AgentState):
    print("[Executor] Running the code...")
    code = state["current_code"]
    
    # We capture standard output (print statements) to show success
    old_stdout = sys.stdout
    redirected_output = sys.stdout = io.StringIO()
    
    error_msg = ""
    
    try:
        # EXEC WARNING: In a real enterprise system, run this inside a Docker sandbox!
        # For a local portfolio piece, executing simple logic scripts is fine.
        exec(code, {})
        output = redirected_output.getvalue()
        print(f"[Executor] Success! Output:\n{output.strip()}")
    except Exception as e:
        # If it crashes, capture the EXACT error traceback to feed back to the LLM
        error_msg = traceback.format_exc()
        print(f"[Executor] Code failed with error: {str(e)}")
    finally:
        # Restore normal print functionality
        sys.stdout = old_stdout
        
    return {"error": error_msg}

# 4. The Router (Conditional Logic)
def route_next_step(state: AgentState):
    if state["error"] == "" and state["iterations"] > 0:
        return "success"
    if state["iterations"] >= 3: # Prevent infinite loops
        return "max_retries"
    return "try_again"

# 5. Build the Graph
workflow = StateGraph(AgentState)

# Add our nodes
workflow.add_node("coder", coder_node)
workflow.add_node("executor", executor_node)

# Connect the flow: Start -> Coder -> Executor -> Evaluate
workflow.set_entry_point("coder")
workflow.add_edge("coder", "executor")

# Add the conditional branching
workflow.add_conditional_edges(
    "executor",
    route_next_step,
    {
        "success": END,
        "max_retries": END,
        "try_again": "coder"
    }
)

# Compile the agent
app = workflow.compile()

# --- RUN THE APP ---
if __name__ == "__main__":
    print("🚀 Starting the Self-Healing Agent...\n")
    
    # The portfolio task
    initial_state = {
    "task": """Create a script that calculates the 'Z-score' for the number 10 in this list: [10, 12, 8, 14, 11]. 
    Use the 'scipy' library for the calculation. 
    If the library is missing, catch the error and rewrite the code to calculate it manually using standard math. 
    Print the final result.""",
    "current_code": "",
    "error": "",
    "iterations": 0
    }

    
    
    # Stream the graph execution
    final_state = app.invoke(initial_state)
    
    print("\n" + "="*50)
    print("🏁 FINAL RESULT")
    print("="*50)
    if final_state["error"] == "":
        print("The agent successfully wrote and validated the code!")
    else:
        print("The agent failed after max retries. Final error:")
        print(final_state["error"])
        
    print("\n--- Final Code ---")
    print(final_state["current_code"])