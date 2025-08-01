import sys
import argparse
from orchestrator import JaraxxusState, decompose_task, execute_next_step, proceed_or_handle_failure, all_steps_done
from orchestrator import reflect_on_failure, formulate_new_plan, invoke_progenitor, integrate_progenitor_suggestion
import memory_manager as mem

parser = argparse.ArgumentParser()
parser.add_argument("--recover", action="store_true", help="Flag to indicate restarting after a crash")
parser.add_argument("goal", nargs="*", help="Optional user goal for the Orchestrator")
args = parser.parse_args()

# Start or restore session
if args.recover:
    # Recovering from crash: reuse last session ID and state from Scratchpad
    try:
        with open("last_session_id.txt", "r") as f:
            session_id = f.read().strip()
    except FileNotFoundError:
        print("No session ID to recover.", file=sys.stderr)
        sys.exit(1)
    print(f"Recovering session {session_id} after crash...")
    # Initialize memory (if not already done)
    mem.mos_config = mem.mos_config  # ensure memory_os is loaded
    # Fetch last known state from Scratchpad (for simplicity, get all text memories)
    memories = mem.memory_os.get_all(mem_cube_id="scratchpad", user_id=session_id)
    state = JaraxxusState(session_id=session_id)
    # If crash log was recorded, include it
    # (Assume the crash log is the last message in scratchpad)
    if memories and isinstance(memories, list):
        for item in memories:
            if isinstance(item, dict) and item.get("content"):
                content = item["content"]
                if isinstance(content, dict) and content.get("crash_log"):
                    state["crash_log"] = content["crash_log"]
                    state["goal"] = content.get("goal", "")
                    state["plan"] = content.get("current_plan", [])
                    state["current_step"] = content.get("current_step", 0)
                    break
    # If no crash_log found, proceed with an empty crash context
else:
    # New session start
    session_id = mem.start_session()
    # Save session_id to file in case we need to recover
    with open("last_session_id.txt", "w") as f:
        f.write(session_id)
    # Determine goal either from CLI args or prompt user
    goal_text = " ".join(args.goal).strip()
    if not goal_text:
        goal_text = input("Enter the task for Jaraxxus: ").strip()
    state = JaraxxusState(session_id=session_id, goal=goal_text)

# Log initial state (goal) to scratchpad
mem.log_to_scratchpad(session_id, {"session_id": session_id, "goal": state.get("goal", ""), "current_plan": [], "last_action": None})

# If recovering, go directly to reading crash context and reflection
if args.recover:
    state = reflect_on_failure(state)
    state = formulate_new_plan(state)
    # Log the recovery analysis to scratchpad
    mem.log_to_scratchpad(session_id, {"analysis": state.get("analysis", ""), "current_plan": state.get("plan", []), "crash_log": state.get("crash_log", "")})

# Main execution loop
completed = False
recovered = False
while not completed:
    if state.get("plan") is None or state.get("current_step") is None:
        # No plan yet - decompose the goal
        state = decompose_task(state)
        mem.log_to_scratchpad(session_id, {"current_plan": state.get("plan", [])})
        print(f"Plan: {state['plan']}")

    # Execute steps one by one
    state = execute_next_step(state)
    # Log after each step
    log_entry = {
        "last_action": state.get("last_action"),
        "last_action_status": state.get("last_action_status"),
        "error": state.get("error", "")
    }
    if state.get("last_file_diff"):
        log_entry["last_file_diff"] = state["last_file_diff"]
    mem.log_to_scratchpad(session_id, log_entry)
    print(f"Step {state['current_step']} ({state['last_action']}): {state['last_action_status']}")
    if state.get("last_action_result"):
        print(f"Result: {state['last_action_result']}")

    # Check if execution should continue or not
    completed = all_steps_done(state)
    if not completed:
        # Advance to next step if no failure
        state = proceed_or_handle_failure(state)
        if state.get("last_action_status") == "FAILURE":
            # A failure occurred – break to trigger recovery
            completed = True

# If execution loop ended due to failure, attempt recovery
if state.get("last_action_status") == "FAILURE":
    print("Failure detected, invoking recovery protocol...")
    # Reflect and replan
    state = reflect_on_failure(state)
    print("Analysis of failure:", state.get("analysis", ""))
    state = formulate_new_plan(state)
    print("New plan after failure:", state.get("plan", []))
    mem.log_to_scratchpad(session_id, {
        "failure_analysis": state.get("analysis", ""),
        "new_plan": state.get("plan", [])
    })
    recovered = True

# If a new plan was formulated (recovered), execute it as well
if recovered:
    completed = False
    while not completed:
        state = execute_next_step(state)
        mem.log_to_scratchpad(session_id, {
            "last_action": state.get("last_action"),
            "last_action_status": state.get("last_action_status"),
            "error": state.get("error", "")
        })
        print(f"Step {state['current_step']} ({state['last_action']}): {state['last_action_status']}")
        if state.get("last_action_result"):
            print(f"Result: {state['last_action_result']}")
        completed = all_steps_done(state)
        if not completed:
            state = proceed_or_handle_failure(state)
            if state.get("last_action_status") == "FAILURE":
                completed = True
                print("New plan also failed; escalating to Progenitor.")
                state = invoke_progenitor(state)
                print("Progenitor output:", state.get("progenitor_suggestion", "")[:200], "...")
                state = integrate_progenitor_suggestion(state)
                # (In a real-case, we might re-run the step after integrating or restart the system.)

# Upon completion, if a novel solution was found after failure, log it to Codex
if recovered and state.get("last_action_status") == "SUCCESS":
    lesson = {
        "lesson_id": str(uuid.uuid4()),
        "problem_signature": state.get("error", "Unknown error"),
        "failure_pattern": state.get("plan", []),
        "successful_resolution": state.get("plan", []),
        "human_guidance": "none",
        "confidence_score": 0.9
    }
    mem.memory_os.add(messages=[{"role": "system", "content": lesson}], user_id=session_id, mem_cube_id="codex")

print("\nTask completed. Final state:")
print(state)
