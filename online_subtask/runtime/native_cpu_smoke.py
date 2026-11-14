"""CPU catalogue/protocol smoke. No policy load, API, GPU or simulator."""
from multiview_protocol import load_catalog, CATALOG_SHA
from task_contract import TASK, INSTRUCTION, CHECKPOINT_STEP, NATIVE_ACTION_LIMIT
import json

def main():
    catalog=load_catalog()
    assert len(catalog['candidates'])==297
    print(json.dumps(dict(passed=True,task=TASK,instruction=INSTRUCTION,
        checkpoint_step=CHECKPOINT_STEP,SDK_action_limit=NATIVE_ACTION_LIMIT,
        catalog_count=297,catalog_sha256=CATALOG_SHA,
        actual_API_GPU_simulator_called=False)))
if __name__=='__main__':main()
