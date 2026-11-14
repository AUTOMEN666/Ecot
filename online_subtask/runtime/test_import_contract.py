"""Source-resolution fixtures only; no real ML imports, GPU or API."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from import_contract import create_contract, validate_contract, activate, validate_policy_environment


def sources(root):
    parent = root/'training'; sdk = root/'sdk'; vendor = parent/'vendor/openpi'
    (parent/'configs').mkdir(parents=True); (parent/'src').mkdir()
    (vendor/'packages/openpi-client/src/openpi_client').mkdir(parents=True)
    (sdk/'XPolicyLab/client_server/ws').mkdir(parents=True)
    (parent/'configs/train.json').write_text(json.dumps({'vendor_root':str(vendor)}))
    files = [parent/'src/strict_prompt.py', sdk/'XPolicyLab/client_server/ws/model_server.py',
        sdk/'XPolicyLab/client_server/ws/model_client.py',
        vendor/'packages/openpi-client/src/openpi_client/base_policy.py']
    for name in ('models.pi0_config','models.pi0','policies.policy_config','policies.policy',
                 'policies.aloha_policy','shared.normalize','training.config','transforms'):
        files.append(vendor/'src/openpi'/(name.replace('.','/')+'.py'))
    for path in files:
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# synthetic source fixture\n')
    return parent,sdk


class ImportSourceContracts(unittest.TestCase):
    def test_policy_library_file_cannot_override_GPU_credentials_or_source_paths(self):
        self.assertEqual(validate_policy_environment({'LD_LIBRARY_PATH':'/own/lib'}), {'LD_LIBRARY_PATH':'/own/lib'})
        for name in ('CUDA_VISIBLE_DEVICES','ECOT_AUTHORIZED_API_KEY','PYTHONPATH','LD_PRELOAD'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_policy_environment({name:'unapproved'})
    def test_training_vendor_RPC_paths_take_precedence_and_modified_files_fail(self):
        with tempfile.TemporaryDirectory()as tmp:
            parent,sdk=sources(Path(tmp));contract=create_contract(parent,sdk)
            self.assertEqual(validate_contract(contract),contract)
            with patch.object(sys,'path',list(sys.path)):
                receipt=activate(contract,import_modules=False)
                self.assertEqual(receipt['contract_sha256'],contract['contract_sha256'])
                self.assertLess(sys.path.index(str((parent/'vendor/openpi/src').resolve())),len(contract['pythonpath'])+1)
            (parent/'vendor/openpi/src/openpi/policies/policy.py').write_text('# changed\n')
            with self.assertRaises(ValueError):validate_contract(contract)

    def test_cached_foreign_module_is_rejected_instead_of_silently_reused(self):
        with tempfile.TemporaryDirectory()as tmp:
            parent,sdk=sources(Path(tmp));contract=create_contract(parent,sdk)
            foreign=types.ModuleType('strict_prompt');foreign.__file__=str(Path(tmp)/'wrong.py')
            with patch.object(sys,'path',list(sys.path)),patch.dict(sys.modules,{'strict_prompt':foreign}):
                with self.assertRaisesRegex(ValueError,'Wrong actual imported source'):
                    activate(contract,import_modules=False)

    def test_missing_RPC_source_is_rejected_before_any_model_load(self):
        with tempfile.TemporaryDirectory()as tmp:
            parent,sdk=sources(Path(tmp))
            (sdk/'XPolicyLab/client_server/ws/model_server.py').unlink()
            with self.assertRaises(FileNotFoundError):create_contract(parent,sdk)


if __name__=='__main__':unittest.main()
