"""Make an isolated steady model spec from a genuine admitted legacy spec.

Does not load weights, call an API, run a simulator or manufacture acceptance.
The runtime still checks the real export binding and external assets.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from model_server import validate_protocol_spec
from subtask_guard import STEADY_CONFIG_ID, STEADY_NAMESPACE


def convert(spec, audit_path):
    validate_protocol_spec(spec)
    if spec.get('steady_subtask_enabled'):
        raise ValueError('Input must be the original admitted legacy specification')
    if STEADY_NAMESPACE not in Path(audit_path).parts:
        raise ValueError('A new isolated steady audit namespace is required')
    result = deepcopy(spec)
    result.update(config_id=STEADY_CONFIG_ID, protocol_config_id=STEADY_CONFIG_ID,
                  generator_protocol=STEADY_CONFIG_ID, steady_subtask_enabled=True,
                  audit_path=str(audit_path))
    validate_protocol_spec(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-spec', type=Path, required=True)
    parser.add_argument('--audit-path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = convert(json.loads(args.base_spec.read_text()), args.audit_path)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
