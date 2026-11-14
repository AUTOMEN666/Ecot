"""Immutable scientific condition for this independent fresh-base experiment."""
VARIANT = 'A_subtask_online'
PROTOCOL = 'pi05_classify_objects100_subtask_online_native100_v1'
ECOT_VARIANT = 'subtask'
RETAINED_FIELDS = ('subtask',)
EVAL_ECOT = True
DROPOUT_PROBABILITY = 0.0
DROPOUT_MODE = 'none'
PROMPT_VERSION = 'classify_subtask_only_natural_v2'
PARAMETER_POLICY = 'pi05_fullparam_subtask_online_v2'
EVAL_STEPS = (5000,10000,15000,20000)
EVAL_EPISODES = 100
EVAL_SCENE_SEEDS = (0,1,2,3,4)
EVAL_LAYOUT_IDS = tuple(range(20))
