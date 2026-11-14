"""Overlay ACTUALLY executed Subtask text on every25Hz native video boundary.

Frame8 shows the text used for action8; the next replan's text first appears at
frame9. No generated-but-unexecuted proposal is substituted into the timeline.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import textwrap

RUNTIME = Path(__file__).resolve().parents[1] / 'online_subtask/runtime'
sys.path.insert(0, str(RUNTIME))
from task_contract import INSTRUCTION, CHECKPOINT_STEP, NATIVE_ACTION_LIMIT, compose_prompt


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as src:
        for block in iter(lambda: src.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def timeline(path, action_count):
    if type(action_count) is not int or not 1 <= action_count <= NATIVE_ACTION_LIMIT:
        raise ValueError('Actual native action count1..800 required')
    raw = Path(path).read_text()
    if not raw.endswith('\n'):
        raise ValueError('Incomplete actual injection log')
    rows = [json.loads(line) for line in raw.splitlines() if line]
    plans = [row for row in rows if row.get('event') == 'actual_injection']
    actions = [row for row in rows if row.get('event') == 'actual_executed_action_text']
    if ([row.get('execution_frame') for row in plans] != list(range(0, action_count, 8))
            or [row.get('execution_action') for row in actions] != list(range(action_count))):
        raise ValueError('One injection per actual replan and one text per executed action required')
    episode = plans[0]['episode_id']; table = {row['execution_frame']: row for row in plans}
    for row in plans:
        if (row.get('episode_id') != episode or row.get('actual_injected') is not True
                or row.get('checkpoint_step') != CHECKPOINT_STEP
                or set(row.get('context', {})) != {'subtask'}
                or row.get('prompt') != compose_prompt(INSTRUCTION, row['context']['subtask'])
                or hashlib.sha256(row['prompt'].encode()).hexdigest() != row.get('prompt_sha256')):
            raise ValueError('Actual EMA10000 Subtask-only prompt required')
    result = [plans[0]['context']['subtask']]
    for index, row in enumerate(actions):
        plan = table[index // 8 * 8]
        if (row.get('episode_id') != episode or row.get('observed_outcome_frame') != index + 1
                or row.get('injection_execution_frame') != plan['execution_frame']
                or row.get('context') != plan['context'] or row.get('prompt') != plan['prompt']
                or row.get('prompt_sha256') != plan['prompt_sha256']):
            raise ValueError('Action text differs from the actual applied prompt')
        result.append(row['context']['subtask'])
    return result


def render(source_video, injection_log, output_video, receipt_path, *, ffmpeg='ffmpeg', ffprobe='ffprobe'):
    from PIL import Image, ImageDraw, ImageFont
    source_video, output_video = Path(source_video), Path(output_video)
    if output_video.exists() or Path(receipt_path).exists():
        raise ValueError('Fresh video/receipt paths required')
    probe = subprocess.run([ffprobe, '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate,nb_read_frames', '-of', 'json', str(source_video)],
        capture_output=True, text=True, timeout=180, check=True)
    stream = json.loads(probe.stdout)['streams'][0]
    count = int(stream['nb_read_frames']); rate = stream['r_frame_rate'].split('/')
    if (stream['width'], stream['height']) != (640, 480) or float(rate[0]) / float(rate[1]) != 25:
        raise ValueError('Original native640x480/25Hz main recording required')
    labels = timeline(injection_log, count - 1)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 22)
    except OSError:
        font = ImageFont.load_default()
    output_video.parent.mkdir(parents=True, exist_ok=True)
    decoder = subprocess.Popen([ffmpeg, '-nostdin', '-v', 'error', '-xerror', '-i', str(source_video),
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    encoder = subprocess.Popen([ffmpeg, '-nostdin', '-n', '-v', 'error', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', '1280x480', '-r', '25', '-i', 'pipe:0', '-an',
        '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output_video)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    decoded = 0
    try:
        for index in range(count):
            size = 640 * 480 * 3; pieces = []; remaining = size
            while remaining:
                block = decoder.stdout.read(remaining)
                if not block:
                    raise ValueError('Video decode ended before all actual boundaries')
                pieces.append(block); remaining -= len(block)
            canvas = Image.new('RGB', (1280, 480), '#101820')
            canvas.paste(Image.frombytes('RGB', (640, 480), b''.join(pieces)), (0, 0))
            draw = ImageDraw.Draw(canvas)
            draw.text((660, 24), 'Stack Bowls Random | EMA10000', font=font, fill='white')
            draw.text((660, 66), f'Action boundary {index}/{count - 1} | 25 Hz', font=font, fill='white')
            draw.text((660, 120), 'Actually applied Subtask:', font=font, fill='#ffe066')
            y = 162
            for line in textwrap.wrap(labels[index], width=43):
                draw.text((660, y), line, font=font, fill='white'); y += 30
            if y > 450:
                raise ValueError('Subtask does not fit video panel; refuse text truncation')
            encoder.stdin.write(canvas.tobytes()); decoded += 1
        if decoder.stdout.read(1):
            raise ValueError('Video has unaccounted extra frames')
        encoder.stdin.close()
        if decoder.wait(timeout=60) or encoder.wait(timeout=180):
            raise RuntimeError('Actual video decode/encode failed')
    finally:
        for child in (decoder, encoder):
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait()
    encoded_probe = subprocess.run([ffprobe, '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate,nb_read_frames', '-of', 'json', str(output_video)],
        capture_output=True, text=True, timeout=180, check=True)
    encoded_stream = json.loads(encoded_probe.stdout)['streams'][0]
    if ((encoded_stream['width'], encoded_stream['height']) != (1280, 480)
            or encoded_stream['r_frame_rate'] != '25/1' or int(encoded_stream['nb_read_frames']) != count):
        raise ValueError('Encoded video dimensions, timing or frame count differs')
    subprocess.run([ffmpeg, '-nostdin', '-v', 'error', '-xerror', '-i', str(output_video),
        '-f', 'null', '-'], capture_output=True, timeout=180, check=True)
    receipt = {'schema': 'ecot.stack_bowls.actual_text_video.v1', 'fully_decoded': True,
               'encoded_video_fully_decoded': True,
               'frame_count': decoded, 'fps': 25, 'actual_text_timeline_verified': True,
               'source_video': str(source_video.resolve()), 'source_video_sha256': sha(source_video),
               'injection_log': str(Path(injection_log).resolve()), 'injection_log_sha256': sha(injection_log),
               'output_video': str(output_video.resolve()), 'output_video_sha256': sha(output_video),
               'interpolation_or_padding': False, 'model_or_API_called': False}
    with Path(receipt_path).open('x') as out:
        json.dump(receipt, out, indent=2); out.write('\n')
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source-video', 'injection-log', 'output-video', 'receipt-path'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(render(a.source_video, a.injection_log, a.output_video, a.receipt_path)))
