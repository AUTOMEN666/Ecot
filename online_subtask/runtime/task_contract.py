"""One approved Stack Bowls checkpoint and native Random evaluation protocol."""
TRAIN_TASK = 'stack_bowls'
TASK = 'stack_bowls_random'
INSTRUCTION = 'Stack the three bowls together.'
CHECKPOINT_STEP = 10000
TRAINING_BUDGET = 15000
VARIANT = 'stack_bowls_subtask_online'
PROTOCOL_CONFIG_ID = 'stack_bowls_random_mainhist93_tok65536_ema10000_v1'
OUTPUT_NAMESPACE = 'stack_bowls_random_seed1_online30_ema10000_v1'
SCENE_SEED = 1
NOISE_SEED = 42
LAYOUT_IDS = tuple(range(30))
NATIVE_ACTION_LIMIT = 800
EXECUTE_STEPS = 8
LAST_REPLAN_FRAME = NATIVE_ACTION_LIMIT - EXECUTE_STEPS
MAX_PAST_RECORDS = LAST_REPLAN_FRAME // EXECUTE_STEPS
DATASET_MANIFEST_SHA = '726a4941dff3d3fbf8132e25f81df0e5e7b14ba5bc7b9d61bfcd1627416146e1'
CONTEXT_CACHE_SHA = '70658dfcd24eed1548f7ff71511486a93377dad0fe7540d0228fb7ead1936c5f'
NORM_SHA = 'c0431ec0c8c1921b4da3131ee9233b78609a334a48d7eede5724b064a32f443c'
INVENTORY_SHA = '33348bc35c2aefd73bac818694fc824e419c76409c092d66403652d8e9830cc9'
CHECKPOINT_SEAL_SHA = '3fb311f3b65b145e760c23c64d80e9063fa4acd244e18d439d33c7838ba11dbe'


def compose_prompt(instruction, subtask, enabled=True):
    if instruction != INSTRUCTION or enabled is not True:
        raise ValueError('Original Stack Bowls instruction and online Subtask required')
    if (not isinstance(subtask, str) or not subtask.strip()
            or subtask != subtask.strip() or len(subtask) > 1200):
        raise ValueError('Nonempty original Subtask text required')
    return instruction + ' Current subtask: ' + subtask
