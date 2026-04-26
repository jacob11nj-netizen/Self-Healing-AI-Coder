import streamlit as st
import os
import re
import sys
import io
import traceback
from typing import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# Load API keys
load_dotenv()

# --- AGENT LOGIC (Same as your main.py but UI-Ready) ---
class AgentState(TypedDict):
    task: str
    current_code: str
    error: str
    iterations: int

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)

def coder_node(state: AgentState):
    with st.status(f"🛠️ Coder: Attempting Version {state['iterations'] + 1}...", expanded=False):
        if state["error"]:
            prompt = f"""You are an expert Python engineer. 
Your previous code failed with this error: {state['error']}

FIX INSTRUCTIONS:
1. Analyze the error traceback provided.
2. Provide a single, corrected version of the script that performs the task.
3. DO NOT include the failing code or 'before/after' comparisons. 
4. Ensure the script is self-contained and executable.

Task: {state['task']}
Output ONLY the raw Python code inside ```python ``` blocks."""
        else:
            prompt = f"""You are an expert Python engineer. 
Write a clean, executable Python script to: {state['task']}
Include print statements to demonstrate the result.
Output ONLY the raw Python code inside ```python ``` blocks."""
        
        response = llm.invoke(prompt)
        code_match = re.search(r"```python\n(.*?)\n```", response.content, re.DOTALL)
        extracted_code = code_match.group(1) if code_match else response.content
        
        st.code(extracted_code, language='python')
        return {"current_code": extracted_code, "iterations": state["iterations"] + 1}
    
def executor_node(state: AgentState):
    with st.status(f"🧪 Executor: Testing Code...", expanded=False):
        code = state["current_code"]
        old_stdout = sys.stdout
        redirected_output = sys.stdout = io.StringIO()
        error_msg = ""
        try:
            exec(code, {})
            output = redirected_output.getvalue()
            st.success("Execution Successful!")
            st.text(f"Output: {output}")
        except Exception as e:
            error_msg = traceback.format_exc()
            st.error(f"Execution Failed: {str(e)}")
        finally:
            sys.stdout = old_stdout
        return {"error": error_msg}

def route_next_step(state: AgentState):
    if state["error"] == "" and state["iterations"] > 0: return "success"
    if state["iterations"] >= 3: return "max_retries"
    return "try_again"

# Build Graph
workflow = StateGraph(AgentState)
workflow.add_node("coder", coder_node)
workflow.add_node("executor", executor_node)
workflow.set_entry_point("coder")
workflow.add_edge("coder", "executor")
workflow.add_conditional_edges("executor", route_next_step, {"success": END, "max_retries": END, "try_again": "coder"})
app = workflow.compile()

# --- STREAMLIT UI ---
st.set_page_config(page_title="Self-Healing AI Coder", page_icon="🤖")

st.title("🤖 Self-Healing AI Engineer")
st.markdown("""
This agent doesn't just write code—it **tests** it in a real Python environment and **fixes** its own bugs autonomously.
""")

with st.sidebar:
    st.header("Settings")
    st.info("Using Llama-3.1 via Groq (High Speed)")
    max_iters = st.slider("Max Retries", 1, 5, 3)

task_input = st.text_area("What should the AI build?", 
                         placeholder="e.g. Write a function to calculate Z-score for [1,2,3] without using scipy.")

if st.button("Run Agent", type="primary"):
    if not task_input:
        st.warning("Please enter a task first.")
    else:
        st.divider()
        initial_state = {
            "task": task_input,
            "current_code": "",
            "error": "",
            "iterations": 0
        }
        
        # Run the graph
        final_state = app.invoke(initial_state)
        
        st.divider()
        if final_state["error"] == "":
            st.balloons()
            st.header("✅ Final Validated Code")
            st.code(final_state["current_code"], language='python')
        else:
            st.error("The agent reached max retries without a valid fix.")
            st.code(final_state["current_code"], language='python')