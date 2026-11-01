#!/usr/bin/env python3
"""FloodNet Track 2 (VQA) -> data/floodnet/questions.jsonl (FLOOD_VLM_PLAN.md 4.2).

Uses the webdataset mirror https://huggingface.co/datasets/ljx620/FloodNet-VQA
(test split: 1,833 Q&A, images resized to 224 x 224 by the mirror).

    ros2 run flood_vlm floodnet_prep --download          # fetch the 2 test shards (~64 MB)
    ros2 run flood_vlm floodnet_prep                     # extract + convert

Output format is shared with FloodSim-VQA:
  {"qid", "dataset", "split", "image", "type", "question", "answer", "answer_type"}
with type in simple_counting | complex_counting | condition | yes_no (+ FloodNet's density).
"""
from __future__ import annotations

import argparse
import json
import os
import tarfile
import urllib.request

BASE = 'https://huggingface.co/datasets/ljx620/FloodNet-VQA/resolve/main/'
SHARDS = ['test/test-00000.tar', 'test/test-00001.tar']

# FloodNet question_type -> our four types (plan 4.1)
TYPE_MAP = {
    'Simple_Counting': 'simple_counting',
    'Complex_Counting': 'complex_counting',
    'Entire_Image_Condition_Recognition': 'condition',
    'Road_Condition_Recognition': 'condition',
    'Yes_No': 'yes_no',
    'Building_Condition_Recognition': 'yes_no',   # "Is the building flooded?" (mirror's names)
    'Risk_Assessment': 'yes_no',
    'Density_Estimation': 'density',
}


def answer_type(ans: str) -> str:
    a = ans.strip().lower()
    if a in ('yes', 'no'):
        return 'yes_no'
    if a.isdigit():
        return 'int'
    return 'label'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw', default='data/floodnet_raw')
    ap.add_argument('--out', default='data/floodnet')
    ap.add_argument('--download', action='store_true')
    args = ap.parse_args(argv)

    if args.download:
        for s in SHARDS:
            dst = os.path.join(args.raw, s)
            if not os.path.exists(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                print('downloading', s)
                urllib.request.urlretrieve(BASE + s, dst)

    img_dir = os.path.join(args.out, 'images')
    os.makedirs(img_dir, exist_ok=True)
    rows = []
    for s in SHARDS:
        path = os.path.join(args.raw, s)
        with tarfile.open(path) as tar:
            members = {m.name: m for m in tar.getmembers() if m.isfile()}
            for name, m in sorted(members.items()):
                if not name.endswith('.json'):
                    continue
                key = name[:-5]
                d = json.load(tar.extractfile(m))
                meta = d['meta']
                q = d['conversations'][0]['value'].replace('<image>\n', '').split('\n')[0].strip()
                a = d['conversations'][1]['value'].strip()
                img_name = meta['image_id']
                img_path = os.path.join(img_dir, img_name)
                if not os.path.exists(img_path):
                    with open(img_path, 'wb') as f:
                        f.write(tar.extractfile(members[key + '.0.img']).read())
                rows.append({'qid': f'fn{meta["question_id"]}', 'dataset': 'floodnet', 'split': 'test',
                             'image': os.path.abspath(img_path),
                             'type': TYPE_MAP.get(meta['question_type'], meta['question_type']),
                             'type_orig': meta['question_type'], 'question': q, 'answer': a.lower(),
                             'answer_type': answer_type(a)})
    out = os.path.join(args.out, 'questions.jsonl')
    with open(out, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    from collections import Counter
    print(f'{len(rows)} questions -> {out}')
    print(Counter(r['type_orig'] for r in rows))


if __name__ == '__main__':
    main()
