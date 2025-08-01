import os
import shutil
import subprocess
import difflib

# Define a safe workspace directory
WORKSPACE_DIR = os.path.abspath("./workspace")
os.makedirs(WORKSPACE_DIR, exist_ok=True)

def _resolve_path(path: str) -> str:
    """Resolve a user-provided path to an absolute path within the workspace."""
    abs_path = os.path.abspath(os.path.join(WORKSPACE_DIR, path))
    # Prevent escaping the workspace
    if not abs_path.startswith(WORKSPACE_DIR):
        raise ValueError(f"Path '{path}' is outside of the workspace.")
    return abs_path

def read_file(path: str) -> str:
    """
    Reads the file at the given path and returns its content as a string.
    :param path: Path to the file (relative to the workspace).
    :return: The content of the file.
    """
    try:
        full_path = _resolve_path(path)
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except Exception as e:
        return f"ERROR: {str(e)}"

def write_file(path: str, content: str) -> str:
    """
    Writes the given content to the file at the given path. Creates the file if it doesn't exist.
    Returns a confirmation message and a diff of changes if the file existed.
    :param path: Path to the file (relative to workspace).
    :param content: Content to write into the file.
    :return: A JSON string with result message and optional diff.
    """
    try:
        full_path = _resolve_path(path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        # Get old content if file exists
        old_content = ""
        if os.path.isfile(full_path):
            with open(full_path, "r", encoding="utf-8") as f:
                old_content = f.read()
        # Write new content
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        result = {"result": "File written successfully."}
        if old_content != "":
            # Compute unified diff
            diff = "".join(difflib.unified_diff(old_content.splitlines(True),
                                                content.splitlines(True),
                                                fromfile="old", tofile="new"))
            result["diff"] = diff
        return result  # This will be returned as a Python dict; LangChain will serialize it to JSON
    except Exception as e:
        return {"error": str(e)}

def list_dir(path: str = ".") -> list:
    """
    Lists files and directories at the given path (relative to workspace).
    :param path: Directory path (defaults to root of workspace).
    :return: List of names in the directory.
    """
    try:
        dir_path = _resolve_path(path)
        items = os.listdir(dir_path)
        return items
    except Exception as e:
        return [f"ERROR: {str(e)}"]

def run_bash_command(command: str) -> dict:
    """
    Executes a bash shell command and returns its output.
    The command is run in the workspace directory.
    :param command: The bash command to execute.
    :return: A dictionary containing "stdout", "stderr", and "exit_code".
    """
    try:
        # Only execute in the workspace for safety
        result = subprocess.run(command, shell=True, cwd=WORKSPACE_DIR,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
        return {
            "stdout": result.stdout[-10000:],  # limit output size if needed
            "stderr": result.stderr[-10000:],
            "exit_code": result.returncode
        }
    except Exception as e:
        # If something goes wrong (e.g., command times out)
        return {
            "stdout": "",
            "stderr": str(e),
            "exit_code": -1
        }
