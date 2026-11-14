"""Original100 numeric data with exact, independently scheduled ECoT conditioning.

The condition mask is a pure function of independent seed/domain/epoch. It never
consumes model, action-noise or sampler RNG, and is independent of worker order.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import threading
import numpy as np
from cover_dataset import CoverBlocksDataset, validate_data_contract
from experiment_settings import (VARIANT, PROTOCOL, ECOT_VARIANT, RETAINED_FIELDS,
    DROPOUT_PROBABILITY, PROMPT_VERSION)

TRAIN_SAMPLES=69562
MASK_DOMAIN=(0x45434F54,0x4245524E)
MASK_SCHEME='all_retained_v1' if DROPOUT_PROBABILITY==0 else 'independent_bernoulli_per_epoch_v1'
SSR_WRAPPER=('\nCurrent-frame action summary, subtask, and reasoning annotations: '
    'use the current observations, task, and supplied annotations. Missing or null fields are unspecified. '
    'No text response is requested.\nContext: ')

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def validate_manifest(manifest):
    validate_data_contract(manifest)
    if (manifest.get('complete')is not True or manifest.get('task_name')!='classify_objects'
        or manifest['data_contract']['episode_indices']!=list(range(200,300))
        or manifest.get('horizon')!=50 or manifest.get('lengths')!={'train':TRAIN_SAMPLES,'val':0}
        or sum(e['valid_action_count']for e in manifest['episodes'])!=TRAIN_SAMPLES):
        raise ValueError('Same original100/all69562 starts/native50 data required')
    return manifest

def source_context_cache_path(context_manifest):
    p=Path(context_manifest);m=json.loads(p.read_text())
    return (Path(m['cache_path'])if m.get('cache_path')else p.parent/Path(m['cache']).name).resolve(strict=True)

def condition_mask(size,epoch,seed=42):
    if type(size)is not int or size<=0 or type(epoch)is not int or epoch<0 or seed!=42:
        raise ValueError('Positive size, explicit nonnegative integer epoch and seed42 required')
    if DROPOUT_PROBABILITY==0.0:mask=np.ones(size,dtype=bool)
    elif DROPOUT_PROBABILITY==0.5:
        rng=np.random.default_rng(np.random.SeedSequence([seed,epoch,*MASK_DOMAIN]))
        mask=rng.random(size)>=DROPOUT_PROBABILITY
    else:raise ValueError('Only the source-bound no-dropout or Bernoulli0.5 variant is permitted')
    mask.setflags(write=False)
    return mask

def ordered_context(context):
    if not isinstance(context,dict)or set(context)!={'subtask','action','reasoning'}:
        raise ValueError('Expected exactly the original three fields')
    if not isinstance(context['action'],dict)or set(context['action'])!={'summary'}:
        raise ValueError('Expected action.summary only')
    values=[context['subtask'],context['action']['summary'],context['reasoning']]
    if any(v is not None and not isinstance(v,str)for v in values):raise ValueError('Original text/null only')
    if not isinstance(context['subtask'],str)or not context['subtask'].strip():raise ValueError('Original Subtask must be nonempty')
    return {'subtask':context['subtask'],'action':{'summary':context['action']['summary']},'reasoning':context['reasoning']}

def compose_prompt(instruction,condition,kept):
    if not isinstance(instruction,str)or not instruction.strip()or type(kept)is not bool:
        raise ValueError('Original instruction and explicit condition branch required')
    if ECOT_VARIANT=='subtask':
        if not isinstance(condition,str)or not condition.strip():raise ValueError('Original Subtask text required')
        return instruction+' Current subtask: '+condition if kept else instruction
    if ECOT_VARIANT!='summary_subtask_reasoning':raise ValueError('Unknown scientific input variant')
    context=ordered_context(condition)
    return instruction+SSR_WRAPPER+json.dumps(context,ensure_ascii=False,separators=(',',':'),allow_nan=False)if kept else instruction

def batch_condition_counts(epoch,batch_index,seed=42,batch_size=32):
    if type(batch_index)is not int or not 0<=batch_index<2174 or seed!=42 or batch_size!=32:
        raise ValueError('Original seed42/GB32 finite sampler batch required')
    order=np.arange(TRAIN_SAMPLES,dtype=np.int64)
    np.random.default_rng(np.random.SeedSequence([seed,epoch])).shuffle(order)
    indices=order[batch_index*batch_size:(batch_index+1)*batch_size]
    kept=int(condition_mask(TRAIN_SAMPLES,epoch,seed)[indices].sum())
    return {'ecot_kept_samples':kept,'ecot_dropped_samples':len(indices)-kept,
            'ecot_padding_excluded':batch_size-len(indices),'real_samples':len(indices)}

class SubtaskDataset(CoverBlocksDataset):
    """Compatibility class name; B preserves all3 fields rather than a Subtask projection."""
    epoch_conditioning=True
    def __init__(self,manifest_path,context_manifest,split='train',horizon=50,dropout_seed=42):
        if split!='train'or horizon!=50 or dropout_seed!=42:raise ValueError('Same original100/train/native50/seed42 required')
        validate_manifest(json.loads(Path(manifest_path).read_text()))
        super().__init__(manifest_path,split=split,horizon=horizon)
        self.variant=VARIANT;self.context_manifest=Path(context_manifest)
        source=json.loads(self.context_manifest.read_text())
        if (source.get('complete')is not True or source.get('counts',{}).get('training_starts')!=TRAIN_SAMPLES
            or source.get('baseline_manifest_sha256')not in(None,file_sha(manifest_path))):
            raise ValueError('Actual original source100 context manifest required')
        raw=source_context_cache_path(context_manifest).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=source['cache_sha256']:raise ValueError('Original annotation cache changed')
        self._conditions={}
        for line in raw.splitlines():
            row=json.loads(line);key=(row['episode_index'],row['frame_index']);context=ordered_context(json.loads(row['context']))
            if key in self._conditions:raise ValueError('Duplicate source frame')
            self._conditions[key]=context['subtask']if ECOT_VARIANT=='subtask'else context
        expected={(ep['episode_index'],i)for ep in self.episodes for i in range(ep['valid_action_count'])}
        if set(self._conditions)!=expected or len(self)!=TRAIN_SAMPLES:raise ValueError('Original condition projection does not cover all starts')
        digest=hashlib.sha256()
        for key in sorted(self._conditions):
            row={'episode_index':key[0],'frame_index':key[1],'condition':self._conditions[key]}
            digest.update((json.dumps(row,sort_keys=True,ensure_ascii=False,separators=(',',':'))+'\n').encode())
        self.projection_sha256=digest.hexdigest();self._masks={};self._mask_lock=threading.Lock()
    def mask(self,epoch):
        if type(epoch)is not int or epoch<0:raise ValueError('Explicit epoch>=0 required')
        with self._mask_lock:
            if epoch not in self._masks:self._masks[epoch]=condition_mask(len(self),epoch)
            return self._masks[epoch]
    def mask_sha256(self,epoch):return hashlib.sha256(self.mask(epoch).tobytes()).hexdigest()
    def prompt_at(self,index,*,epoch=None,kept=None):
        ei,frame=self.locate(index);ep=self.episodes[ei]
        if kept is None:kept=bool(self.mask(epoch)[index])
        return compose_prompt(ep['instruction'],self._conditions[(ep['episode_index'],frame)],kept),kept
    def get_raw(self,index,*,epoch=None,kept=None):
        sample=super().get_raw(index);prompt,actual=self.prompt_at(index,epoch=epoch,kept=kept)
        sample['prompt']=prompt;sample['ecot_condition_kept']=actual;return sample
    __getitem__=get_raw
