"""One actual original A5000 CPU inference using real saved frame0/real selector output.

No simulator, native commands, CUDA, credential, new generation or answer fallback.
Root supplies an actual saved selector's strict single-ID response, never a GT ID.
"""
from __future__ import annotations
import argparse,base64,hashlib,io,json,os,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--request',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=''or os.environ.get('JAX_PLATFORMS')!='cpu':raise ValueError('Explicit hiddenCUDA and CPU-only JAX required')
    a.output.mkdir(parents=True,exist_ok=False)
    from multiview_protocol import CAMERAS,INSTRUCTION,resolve,strict_json,sha
    request=strict_json(a.request.read_bytes());selection=a.selection.read_bytes();context,resolution=resolve(selection)
    parts=request['messages'][-1]['content'];header=strict_json(parts[0]['text'])
    if header['instruction']!=INSTRUCTION or header['current_frame']!=0:raise ValueError('Original frame0/Instruction required; no frame relabeling')
    from PIL import Image
    import numpy as np
    from history_client import decode_png_chw
    images={};states=[];proof=[]
    for i in range(1,len(parts),2):
        meta=strict_json(parts[i]['text'])
        if meta['frame']!=0:continue
        camera=meta['camera'];raw=base64.b64decode(parts[i+1]['image_url']['url'].split(',',1)[1],validate=True)
        im=Image.open(io.BytesIO(raw));im.load()
        if camera not in CAMERAS or camera in images or im.mode!='RGB'or im.size!=(640,480):raise ValueError('Original current3 RGB frame0 required')
        images[camera]=np.asarray(im);states.append(meta['state14']);proof.append({'camera':camera,'png_sha256':sha(raw),'RGB_sha256':sha(images[camera].tobytes()),
            'CHW_sha256':sha(np.ascontiguousarray(images[camera].transpose(2,0,1)).tobytes())})
    if set(images)!=set(CAMERAS)or any(s!=states[0]for s in states):raise ValueError('Matched three views/State14 required')
    class ActualSavedSelectionReplay:
        def reset(self,episode_id):self.episode_id=episode_id
        def generate(self,pngs,instruction,*,state14,episode_id,frame):
            if frame!=0 or state14!=states[0]or instruction!=INSTRUCTION:raise ValueError('Saved model-output replay binding differs')
            expected={x['camera']:x['CHW_sha256']for x in proof}
            if set(pngs)!=set(expected)or any(sha(decode_png_chw(pngs[camera]))!=expected[camera]for camera in CAMERAS):
                raise ValueError('Actual NativeModel generation PNG pixels differ from saved model request')
            return context,{'generator_protocol':'saved_actual_selector_output_for_CPU_policy_smoke_only',
                'request_sha256':sha(a.request.read_bytes()),'actual_saved_selection_sha256':sha(selection),'resolution':resolution,
                'generation_total_seconds':0.0,'new_generation':False,'GT_used':False,'fallback_used':False,'actual_generator_CHW_matches_saved_request':True}
    spec=strict_json(a.spec.read_bytes());spec['audit_path']=str(a.output/'actions.jsonl')
    # Admission requires the explicit isolated namespace in the output path.
    if 'online96_10hz_20261112'not in a.output.resolve().parts:raise ValueError('Explicit isolated comparison output required')
    from model_server import create_model
    import jax
    if jax.default_backend()!='cpu'or any(d.platform!='cpu'for d in jax.devices()):raise RuntimeError('JAX initialized a non-CPU device')
    started=time.monotonic();model=create_model(spec,provider=ActualSavedSelectionReplay());loaded=time.monotonic()
    model.reset();model.update_obs({'instruction':INSTRUCTION,'images':images,'state':states[0],'frame_index':0,'physics_clock':None})
    commands=model.get_action();ended=time.monotonic()
    if len(commands)!=8:raise ValueError('Original to_commands must return exactly8 actions')
    arrays=[{k:np.asarray(v).tolist()for k,v in command.items()}for command in commands]
    result={'passed':True,'actual_original_policy_loaded':True,'backend':jax.default_backend(),'devices':[str(d)for d in jax.devices()],
        'model_load_seconds':loaded-started,'one_policy_infer_seconds':ended-loaded,'decoded_command_count':8,'commands':arrays,
        'source_request_sha256':sha(a.request.read_bytes()),'saved_actual_selection_sha256':sha(selection),'selected_context':context,
        'current_images':proof,'new_native_actions':0,'new_selector_generations':0,'scope':'CPU policy/decoding smoke only; saved real selector output replay; not a native rollout'}
    (a.output/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k]for k in ['passed','backend','model_load_seconds','one_policy_infer_seconds','decoded_command_count','new_native_actions']}),flush=True)
if __name__=='__main__':main()
