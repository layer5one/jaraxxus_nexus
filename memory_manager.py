import uuid
from memos.configs.mem_os import MOSConfig
from memos.mem_os.main import MOS

# Load MemOS configuration and connect to the memory server
mos_config = MOSConfig.from_json_file("memos_config.json")
memory_os = MOS(mos_config)

# Ensure global Codex exists (create once if not already created)
GLOBAL_CODex_USER = "jaraxxus_global"
try:
    memory_os.create_user(user_id=GLOBAL_CODex_USER)
except Exception:
    pass  # user may already exist
# Create the Codex memcube under the global user if not exists
try:
    memory_os.create_cube_for_user(cube_name="codex", owner_id=GLOBAL_CODex_USER)
except Exception:
    pass  # codex may already exist

def start_session():
    """Initialize a new Jaraxxus session with its own Scratchpad memory."""
    session_id = str(uuid.uuid4())
    memory_os.create_user(user_id=session_id)
    # Create a fresh Scratchpad cube for this session
    memory_os.create_cube_for_user(cube_name="scratchpad", owner_id=session_id)
    # Share the global Codex with this session for read/write access
    memory_os.share_cube_with_user(cube_id=f"{GLOBAL_CODex_USER}/codex", target_user_id=session_id)
    return session_id

def log_to_scratchpad(session_id, state: dict):
    """Write the current state (or partial state) to the Scratchpad MemCube."""
    # Prepare the memory item as a structured JSON content
    scratchpad_entry = {
        "role": "system",
        "content": state  # state is already a dict of fields like goal, last_action, etc.
    }
    memory_os.add(messages=[scratchpad_entry], user_id=session_id, mem_cube_id="scratchpad")

def search_codex(session_id, query: str):
    """Search the Codex MemCube for relevant past lessons."""
    results = memory_os.search(query=query, user_id=session_id, install_cube_ids=["codex"])
    return results.get("text_mem", [])
