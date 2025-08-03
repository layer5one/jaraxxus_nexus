import uuid
from memos.mem_os.main import MOS
from memos.configs.mem_os import MOSConfig

mos_config = MOSConfig.from_json_file("examples/data/config/simple_memos_config.json")

mos = MOS(mos_config)

# Generate a unique user ID
user_id = str(uuid.uuid4())

# Create the user
mos.create_user(user_id=user_id)

# Register a simple memory cube for this user
mos.register_mem_cube("examples/data/mem_cube_2", user_id=user_id)
