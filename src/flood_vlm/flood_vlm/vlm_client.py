"""OpenAI-style client for llama-server (FLOOD_VLM_PLAN.md 2.1).

    client = VLMClient(url, run_dir='runs/2026-11-01_1200')
    out, timings = client.ask(img_bgr, P_SURVEY, SURVEY, think=False, max_tokens=384, prompt_id='survey')

Output is constrained with response_format json_schema. On a parse failure the call
is retried once at temperature 0; after that `out` is {"error": ...}.
Every call is appended to <run_dir>/vlm_calls.jsonl.
"""
from __future__ import annotations

import base64
import json
import os
import threading
import time
import urllib.error
import urllib.request

import cv2
import numpy as np


def encode_jpeg(img: np.ndarray, quality: int = 90, long_side: int | None = None) -> bytes:
    if long_side and max(img.shape[:2]) != long_side:
        s = long_side / max(img.shape[:2])
        img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError('JPEG encode failed')
    return buf.tobytes()


def parse_json(text: str) -> dict:
    """Parse a JSON object, tolerating code fences or text around it."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    a, b = text.find('{'), text.rfind('}')
    if a >= 0 and b > a:
        return json.loads(text[a:b + 1])
    raise json.JSONDecodeError('no JSON object', text, 0)


class VLMClient:
    def __init__(self, url: str = 'http://127.0.0.1:8080/v1/chat/completions',
                 run_dir: str | None = None, temperature: float = 0.2,
                 image_long_side: int | None = None, timeout_s: float = 1800.0,
                 schema_mode: str = 'json_schema'):
        self.url = url
        self.run_dir = run_dir
        self.temperature = temperature
        self.image_long_side = image_long_side
        self.timeout_s = timeout_s
        self.schema_mode = schema_mode          # json_schema | json_object | none
        self.n_calls = 0
        self._lock = threading.Lock()
        if run_dir:
            os.makedirs(os.path.join(run_dir, 'frames'), exist_ok=True)

    # ── low level ─────────────────────────────────────────────────────────────
    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(self.url, data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            return json.loads(r.read())

    def _payload(self, image_jpeg: bytes | None, prompt: str, schema: dict | None,
                 think: bool, max_tokens: int, temperature: float) -> dict:
        content = []
        if image_jpeg is not None:
            b64 = base64.b64encode(image_jpeg).decode()
            content.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}})
        content.append({'type': 'text', 'text': prompt})
        payload = {
            'messages': [{'role': 'user', 'content': content}],
            'max_tokens': max_tokens,
            'temperature': temperature,
            'chat_template_kwargs': {'enable_thinking': bool(think)},
        }
        if schema is not None and self.schema_mode == 'json_schema':
            payload['response_format'] = {'type': 'json_schema',
                                          'json_schema': {'name': 'out', 'schema': schema}}
        elif schema is not None and self.schema_mode == 'json_object':
            payload['response_format'] = {'type': 'json_object'}
        return payload

    # ── public ────────────────────────────────────────────────────────────────
    def ask(self, image: np.ndarray | None, prompt: str, schema: dict | None = None,
            think: bool = False, max_tokens: int = 384, prompt_id: str = '',
            image_path: str | None = None, meta: dict | None = None) -> tuple[dict, dict]:
        """Returns (parsed JSON or {"error": ...}, timings incl. wall_s and n_tries)."""
        jpeg = None
        if image is not None:
            jpeg = encode_jpeg(image, 90, self.image_long_side)
            if image_path is None and self.run_dir:
                image_path = os.path.join(self.run_dir, 'frames', f'{self.n_calls:04d}_{prompt_id}.jpg')
                with open(image_path, 'wb') as f:
                    f.write(jpeg)
        t0 = time.time()
        out, timings, raw, reasoning, err = None, {}, '', '', None
        tries = 0
        for temp in (self.temperature, 0.0):
            tries += 1
            try:
                resp = self._post(self._payload(jpeg, prompt, schema, think, max_tokens, temp))
                msg = resp['choices'][0]['message']
                raw = msg.get('content') or ''
                reasoning = msg.get('reasoning_content') or ''
                timings = resp.get('timings', {})
                timings['finish_reason'] = resp['choices'][0].get('finish_reason')
                out = parse_json(raw)
                err = None
                break
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                err = f'parse: {e}'
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                err = f'http: {e}'
                break
        if out is None:
            out = {'error': err, 'raw': raw[:2000]}
        wall = time.time() - t0
        timings = dict(timings, wall_s=round(wall, 2), n_tries=tries)
        self._log(prompt_id, image_path, prompt, out, timings, think, reasoning, meta)
        return out, timings

    def _log(self, prompt_id, image_path, prompt, out, timings, think, reasoning, meta):
        with self._lock:
            self.n_calls += 1
            if not self.run_dir:
                return
            rec = {'i': self.n_calls, 't': time.time(), 'prompt_id': prompt_id, 'image': image_path,
                   'think': think, 'prompt': prompt if image_path is None else prompt[:400],
                   'output': out, 'timings': timings, 'reasoning': reasoning[:4000]}
            if meta:
                rec['meta'] = meta
            with open(os.path.join(self.run_dir, 'vlm_calls.jsonl'), 'a') as f:
                f.write(json.dumps(rec) + '\n')

    def healthy(self) -> bool:
        try:
            with urllib.request.urlopen(self.url.replace('/v1/chat/completions', '/health'), timeout=3) as r:
                return r.status == 200
        except Exception:
            return False
