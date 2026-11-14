"""Lossless current RGB and one-field context helpers. No API entry point."""
from __future__ import annotations
import hashlib,json,struct,zlib

def sha(raw):return hashlib.sha256(raw).hexdigest()

def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def strict_json(raw):
    def pairs(items):
        value={}
        for k,v in items:
            if k in value:raise ValueError('Duplicate JSON key')
            value[k]=v
        return value
    def reject(x):raise ValueError('Nonfinite JSON')
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=reject)

def validate_context(value):
    if (not isinstance(value,dict)or set(value)!={'subtask'}or not isinstance(value['subtask'],str)
            or not value['subtask'].strip()or value['subtask']!=value['subtask'].strip()
            or len(value['subtask'])>1200):
        raise ValueError('Expected only one nonempty, trimmed subtask string <=1200 characters')
    return value

def png_current_rgb(chw):
    import numpy as np
    a=np.asarray(chw)
    if a.shape!=(3,480,640)or a.dtype!=np.uint8:raise ValueError('Actual uint8 CHW current main image required')
    rgb=np.ascontiguousarray(a.transpose(1,2,0))
    def chunk(kind,data):return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
    # PNG Up/filter2 is an exact modulo256 difference against the preceding
    # original RGB row. Only wire compression changes; every pixel is retained.
    filtered=rgb.copy()
    np.subtract(rgb[1:],rgb[:-1],out=filtered[1:])
    raw=b''.join(b'\2'+row.tobytes()for row in filtered)
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',640,480,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw,9))+chunk(b'IEND',b'')
