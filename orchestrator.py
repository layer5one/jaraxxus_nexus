from typing import TypedDict, List, Optional
from langchain_ollama.chat_models import ChatOllama
from langchain.agents import initialize_agent, AgentType

# Import the tools and memory manager
from tools import read_file, write_file, list_dir, run_bash_command
import memory_manager as mem

# Define the structure of the state used in the LangGraph
class JaraxxusState(TypedDict, total=False):
    session_id: str
    goal: str
    plan: List[str]
    current_step: int
    last_action: Optional[str]
    last_action_result: Optional[str]
    last_action_status: Optional[str]  # "SUCCESS" or "FAILURE"
    last_file_diff: Optional[str]
    error: Optional[str]         # error message if any
    crash_log: Optional[str]     # crash log if recovering from crash

# Initialize LLMs for Orchestrator and Specialists via LangChain-Ollama
orchestrator_llm = ChatOllama(model="gemma3:12b-it-qat", temperature=0.7)
file_io_llm = ChatOllama(model="jaraxxus-file-agent", temperature=0.2)
code_exec_llm = ChatOllama(model="jaraxxus-code-agent", temperature=0.1)
data_analysis_llm = ChatOllama(model="jaraxxus-data-agent", temperature=0.3)

# Bind appropriate tools to each specialist LLM
file_io_agent = file_io_llm.bind_tools([read_file, write_file, list_dir])
code_exec_agent = code_exec_llm.bind_tools([run_bash_command])
data_agent = data_analysis_llm  # no tools needed for data agent in this context

def decompose_task(state: JaraxxusState) -> JaraxxusState:
    """Orchestrator LLM: Decompose the user's goal into a plan (list of steps)."""
    goal = state["goal"]
    prompt = (
        f"You are the Orchestrator. Decompose the following user goal into a sequence of steps.\n"
        f"Goal: {goal}\n"
        "Provide a numbered list of concise steps to achieve this goal."
    )
    response = orchestrator_llm.predict(prompt=prompt)
    # Extract plan as a list of step descriptions (split by lines or numbers)
    plan_steps = [line.strip() for line in response.splitlines() if line.strip()]
    # Remove any numeric prefixes from steps:
    plan_steps = [step.split('.', 1)[-1].strip() for step in plan_steps]
    state["plan"] = plan_steps
    state["current_step"] = 0
    return state

def execute_next_step(state: JaraxxusState) -> JaraxxusState:
    """Execute the next planned step by delegating to the appropriate Specialist."""
    idx = state["current_step"]
    plan = state.get("plan", [])
    if idx is None or idx >= len(plan):
        return state  # no more steps
    step_description = plan[idx]
    state["last_action"] = step_description
    state["last_action_result"] = ""
    state["last_action_status"] = ""

    # Determine which specialist to invoke based on the step description
    result = None
    try:
        if any(kw in step_description.lower() for kw in ["file", "read", "write", "open", "save"]):
            # File-related step -> File I/O agent
            # Provide step description as the "user" prompt for the agent
            result_msg = file_io_agent.invoke({"role": "user", "content": step_description})
            result = result_msg.content if hasattr(result_msg, "content") else str(result_msg)
        elif any(kw in step_description.lower() for kw in ["shell", "command", "execute", "run"]):
            # Code execution step -> Code Exec agent
            result_msg = code_exec_agent.invoke({"role": "user", "content": step_description})
            result = result_msg.content if hasattr(result_msg, "content") else str(result_msg)
        else:
            # Default: use Data Analysis agent for analytical/summarization steps
            # If we have some context (like content from a previous file read) to include,
            # it should be appended to the prompt.
            prompt = step_description
            # If the previous step was a file read, include that content for summarization
            if state.get("last_action") and "read" in state["last_action"].lower() and state.get("last_action_result"):
                prompt += "\nContent:\n" + state["last_action_result"]
            result_msg = data_agent.predict(prompt=prompt)
            result = result_msg if isinstance(result_msg, str) else result_msg.content

        # Record the result
        state["last_action_result"] = result
        # Determine success/failure
        if isinstance(result, str) and result.strip().startswith("ERROR"):
            state["last_action_status"] = "FAILURE"
            state["error"] = result
        elif isinstance(result, str) and "\"exit_code\":" in result:
            # If output contains exit_code, inspect it (assume JSON in string)
            try:
                import json
                out_json = json.loads(result)
                exit_code = out_json.get("exit_code", 0)
            except Exception:
                exit_code = 0
            state["last_action_status"] = "FAILURE" if exit_code and exit_code != 0 else "SUCCESS"
            if state["last_action_status"] == "FAILURE":
                state["error"] = f"Command returned exit code {exit_code}"
        else:
            state["last_action_status"] = "SUCCESS"
        # If write_file provided a diff in dict form, extract to state for logging
        if isinstance(result, dict) and result.get("diff"):
            state["last_file_diff"] = result["diff"]
    except Exception as e:
        # If the agent invocation raised an error (unexpected)
        state["last_action_result"] = ""
        state["last_action_status"] = "FAILURE"
        state["error"] = f"Exception during step: {str(e)}"
    return state

def proceed_or_handle_failure(state: JaraxxusState) -> JaraxxusState:
    """Decide whether to continue to next step or initiate failure recovery."""
    if state.get("last_action_status") == "FAILURE":
        # On failure, we won't increment the step index here (trigger recovery instead)
        return state
    # On success, move to the next step
    state["current_step"] += 1
    return state

def all_steps_done(state: JaraxxusState) -> bool:
    """Check if all plan steps have been executed or if a failure occurred."""
    if state.get("last_action_status") == "FAILURE":
        return True  # terminate the normal flow to switch to recovery
    return state.get("current_step", 0) >= len(state.get("plan", []))

# Recovery nodes
def reflect_on_failure(state: JaraxxusState) -> JaraxxusState:
    """Orchestrator reflects on the failure using its LLM to diagnose and suggest a fix."""
    goal = state.get("goal", "")
    last_action = state.get("last_action", "")
    error = state.get("error", state.get("crash_log", ""))
    prompt = (
        "You are an expert debugging agent. The previous attempt to execute the plan failed.\n"
        f"Original Goal: {goal}\n"
        f"Last Action: {last_action}\n"
        f"Error: {error}\n"
        "Analyze the cause of the failure and suggest a new plan or fix."
    )
    analysis = orchestrator_llm.predict(prompt=prompt)
    state["analysis"] = analysis  # store the reflection analysis (for logging or output)
    return state

def formulate_new_plan(state: JaraxxusState) -> JaraxxusState:
    """Orchestrator formulates a new plan after reflecting on a failure."""
    analysis = state.get("analysis", "")
    prompt = (
        "Given the above analysis of the failure, devise a corrected plan to achieve the goal.\n"
        f"Goal: {state.get('goal')}\n"
        "New Plan:"
    )
    if analysis:
        prompt = analysis + "\n\n" + prompt
    response = orchestrator_llm.predict(prompt=prompt)
    new_plan_steps = [line.strip() for line in response.splitlines() if line.strip()]
    new_plan_steps = [step.split('.', 1)[-1].strip() for step in new_plan_steps]
    state["plan"] = new_plan_steps
    state["current_step"] = 0
    state["last_action"] = None
    state["last_action_status"] = None
    state["error"] = None
    return state

def invoke_progenitor(state: JaraxxusState) -> JaraxxusState:
    """If local agents cannot solve the issue, invoke the Gemini Progenitor for assistance."""
    # Construct a prompt for the Gemini CLI with relevant context:
    problem = state.get("error", "an unresolved problem")
    goal = state.get("goal", "")
    prompt = f"Goal: {goal}\nProblem: {problem}\nProvide guidance or tool code to solve this."
    try:
        # Call Gemini CLI (assuming it's installed and configured)
        result = subprocess.run(
            ["gemini", "ask", "--stdin"], input=prompt, text=True, capture_output=True, timeout=120
        )
        progenitor_output = result.stdout.strip()
    except Exception as e:
        progenitor_output = f"Progenitor invocation failed: {e}"
    state["progenitor_suggestion"] = progenitor_output
    return state

def integrate_progenitor_suggestion(state: JaraxxusState) -> JaraxxusState:
    """Integrate the Progenitor's output - e.g., create a new tool or apply code changes."""
    suggestion = state.get("progenitor_suggestion", "")
    # In a real scenario, we might parse the suggestion (which could be code for a new tool or diff)
    # and apply it. For now, we log it to Codex and mark the task as handled.
    if suggestion:
        # For example, if suggestion contains code for a new tool, we could write it to a file:
        # (This is a placeholder; actual integration would depend on suggestion format)
        with open(os.path.join(WORKSPACE_DIR, "progenitor_suggestion.txt"), "w") as f:
            f.write(suggestion)
    state["last_action_result"] = "Integrated Progenitor's suggestion."
    state["last_action_status"] = "SUCCESS"
    return state
