"""Mission report: report.md, map.png, metrics.json (FLOOD_VLM_PLAN.md 3.2)."""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from flood_vlm import gt as G  # noqa: E402
from flood_vlm import scoring  # noqa: E402

LEVEL_COLOUR = {'critical': '#d62728', 'high': '#ff7f0e', 'medium': '#1f77b4', None: '#7f7f7f'}
STACK_PROCS = ('llama-server', 'gz sim', 'ruby', 'parameter_bridge', 'ekf_node', 'mission_node',
               'robot_state_publisher', 'navsat_transform', 'drone_way_home', 'water_controller')


def stack_memory_mb() -> dict:
    """Peak RSS (VmHWM) of the processes that make up the stack, from /proc."""
    out = {}
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
        try:
            with open(f'/proc/{pid}/cmdline', 'rb') as f:
                cmd = f.read().replace(b'\0', b' ').decode(errors='ignore')
            name = next((n for n in STACK_PROCS if n in cmd), None)
            if name is None:
                continue
            with open(f'/proc/{pid}/status') as f:
                hwm = next(int(line.split()[1]) for line in f if line.startswith('VmHWM'))
            out[name] = out.get(name, 0) + hwm // 1024
        except (OSError, StopIteration, ValueError):
            continue
    out['total'] = sum(out.values())
    return out


def vlm_totals(run_dir: str) -> dict:
    path = os.path.join(run_dir, 'vlm_calls.jsonl')
    calls = []
    if os.path.exists(path):
        with open(path) as f:
            calls = [json.loads(line) for line in f if line.strip()]
    s = lambda k: sum(c['timings'].get(k) or 0 for c in calls)  # noqa: E731
    return {'n_vlm_calls': len(calls), 'n_json_errors': sum('error' in c['output'] for c in calls),
            'prompt_tokens': s('prompt_n'), 'gen_tokens': s('predicted_n'),
            'prompt_ms': round(s('prompt_ms')), 'gen_ms': round(s('predicted_ms')),
            'vlm_wall_s': round(s('wall_s'), 1)}


def _annotate_replans(m, gt: dict, by_det: dict):
    """Mark which detected people became more urgent in GT at each water rise."""
    for ev in m.replans:
        worse = []
        for det_id, g in by_det.items():
            if det_id in ev['done']:
                continue
            old = G.LEVEL_RANK[G.urgency(gt, g, ev['old_water'])]
            new = G.LEVEL_RANK[G.urgency(gt, g, ev['new_water'])]
            if new < old:
                worse.append(det_id)
        ev['gt_newly_urgent'] = worse


def draw_map(m, gt: dict | None, path: str):
    fig, ax = plt.subplots(figsize=(9, 9))
    w = m.io.water
    ax.set_facecolor('#8c7350')                  # flood water
    a = m.cfg['area']
    if gt:
        for z in gt['dry_zones']:
            if z['type'] == 'roof':
                continue
            col = '#6b8e3a' if z['type'] == 'mound' else '#8a8a8a'
            if z['top_z'] <= w:
                col = '#a08a60'
            ax.add_patch(Polygon(z['polygon'], closed=True, fc=col, ec='k', lw=0.5, alpha=0.9))
        for h in gt['houses']:
            col = '#e8e3d6' if h['roof_z'] - w >= 1.0 else '#f2c4a0'
            ax.add_patch(Polygon(h['polygon'], closed=True, fc=col, ec='#444', lw=0.8))
            ax.text(h['x'], h['y'] - h['sy'] / 2 + 0.6, f"{h['id']} {h['roof_z']:.1f} m",
                    ha='center', fontsize=6, color='#333')
        for p in gt['people']:
            ax.plot(p['x'], p['y'], 'o', mfc='none', mec='k', ms=13, mew=1.5)
            ax.text(p['x'] + 0.9, p['y'] - 1.6, p['id'], fontsize=7, color='k')
    if len(m.io.path_world) > 1:
        xs = [q[1] for q in m.io.path_world]
        ys = [q[2] for q in m.io.path_world]
        ax.plot(xs, ys, '-', color='#2ca02c', lw=1.0, alpha=0.8, label='drone path')
    order = {r['person_id']: i + 1 for i, r in enumerate(m.ranking)}
    for p in m.mem.people:
        if p.status == 'rejected':
            ax.plot(p.xyz_world[0], p.xyz_world[1], 'x', color='#555', ms=8)
            continue
        ax.plot(p.xyz_world[0], p.xyz_world[1], 'o', color=LEVEL_COLOUR.get(p.level), ms=9)
        if p.survey_xyz and p.looked:
            ax.plot([p.survey_xyz[0], p.xyz_world[0]], [p.survey_xyz[1], p.xyz_world[1]], ':',
                    color=LEVEL_COLOUR.get(p.level), lw=1)
        ax.text(p.xyz_world[0] - 0.4, p.xyz_world[1] + 0.9, str(order.get(p.id, '?')),
                fontsize=10, weight='bold', color='white',
                bbox=dict(boxstyle='circle,pad=0.15', fc=LEVEL_COLOUR.get(p.level), ec='none'))
    for b in m.beacons:
        ax.plot(b['x'], b['y'], '*', color='#ff8c00', mec='k', ms=16)
    h = m.cfg['home']
    ax.plot(h['x'], h['y'], 's', color='white', mec='k', ms=10)
    ax.text(h['x'] + 1.2, h['y'], 'home', fontsize=8)
    for lvl in ('critical', 'high', 'medium'):
        ax.plot([], [], 'o', color=LEVEL_COLOUR[lvl], label=lvl)
    ax.plot([], [], 'o', mfc='none', mec='k', label='ground truth')
    ax.plot([], [], '*', color='#ff8c00', mec='k', ms=12, label='beacon')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax.set_xlim(a['xmin'] - 2, a['xmax'] + 2)
    ax.set_ylim(a['ymin'] - 2, a['ymax'] + 2)
    ax.set_aspect('equal')
    ax.set_xlabel('world x (m)')
    ax.set_ylabel('world y (m)')
    ax.set_title(f'FloodScout-VLM seed {m.args["seed"]} - water {w:.1f} m - numbers = rescue order')
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_all(m, gt: dict | None):
    rd = m.run_dir
    people = m.mem.to_dict()['people']
    metrics = {'run_id': m.run_id, 'seed': m.args['seed'], 'lookagain': m.args.get('lookagain', True),
               'water_final': m.io.water, 'plan_source': m.plan_info.get('source'),
               'plan_valid_first_try': m.plan_info.get('valid_first_try')}
    if gt:
        det = [p for p in people if p['status'] != 'rejected']
        ds = scoring.detection_scores([p['xyz_world'][:2] for p in det], gt['people'])
        by_det = {det[i]['id']: gt['people'][j] for i, j, _ in ds['matches']}
        _annotate_replans(m, gt, by_det)
        metrics.update(scoring.score_mission(gt, people, m.ranking, m.beacons, m.replans,
                                             m.ranking_water if m.ranking_water is not None else m.io.water))
        metrics['beacon_checks'] = []
        for b in m.beacons:
            g = by_det.get(b['person_id'])
            ok, why = G.valid_beacon(gt, g, b['x'], b['y'], b['water']) if g else (False, 'no GT match')
            metrics['beacon_checks'].append({'person_id': b['person_id'], 'gt': g['id'] if g else None,
                                             'safe': ok, 'why': why})
    metrics.update(vlm_totals(rd))
    metrics.update({'vlm_s': round(m.t_vlm, 1), 'flight_sim_s': round(m.t_flight_sim, 1),
                    'flight_wall_s': round(m.t_flight_wall, 1),
                    'total_wall_s': round(getattr(m, 't_total_wall', 0.0), 1),
                    'n_replans': len(m.replans), 'water_log': m.water_log,
                    'ram_peak_mb': stack_memory_mb()})
    with open(os.path.join(rd, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=1)
    with open(os.path.join(rd, 'memory.json'), 'w') as f:
        json.dump({'people': people, 'ranking': m.ranking, 'beacons': m.beacons, 'replans': m.replans,
                   'plan': m.plan_info}, f, indent=1)
    draw_map(m, gt, os.path.join(rd, 'map.png'))

    # ── report.md ─────────────────────────────────────────────────────────────
    L = [f'# FloodScout-VLM report — {m.run_id}', '',
         f'**Mission:** "{m.events[0]["msg"].split(": ", 1)[1].strip(chr(34)) if m.events else ""}"', '',
         f'Seed {m.args["seed"]}, look-again {"on" if m.args.get("lookagain", True) else "off"}, '
         f'water now {m.io.water:.1f} m.', '', '## Plan', '']
    pi = m.plan_info
    L.append(f'Source: **{pi.get("source")}**, valid on first try: **{pi.get("valid_first_try")}**'
             + (f' (errors: {pi.get("errors_first")})' if pi.get('errors_first') else ''))
    L.append('')
    for i, s in enumerate(pi.get('plan', {}).get('steps', [])):
        L.append(f'{i + 1}. `{s["tool"]}` {json.dumps(s["args"]) if s["args"] else ""}')
    L += ['', '## Rescue list (in order)', '', '| # | person | level | posture / location | position (m) | beacon | reason |',
          '|---|---|---|---|---|---|---|']
    beacon_of = {b['person_id']: b for b in m.beacons}
    for i, r in enumerate(m.ranking):
        p = m.mem.get(r['person_id'])
        b = beacon_of.get(p.id)
        L.append(f'| {i + 1} | {p.id} | {r["level"]} | {p.posture} / {p.location} | '
                 f'({p.xyz_world[0]:.1f}, {p.xyz_world[1]:.1f}) | '
                 f'{f"({b[chr(120)]:.1f}, {b[chr(121)]:.1f})" if b else "-"} | {r["reason"]} |')
    L += ['', '### Crops', '']
    for r in m.ranking:
        p = m.mem.get(r['person_id'])
        if p.crops:
            L.append(f'**{p.id}** ' + ' '.join(f'![{p.id}]({os.path.relpath(c, rd)})' for c in p.crops[-2:]))
    L += ['', '## Water and re-planning', '']
    for t, lvl in m.water_log:
        L.append(f'- t={t:.0f} s: water {lvl:.1f} m')
    for e in m.events:
        if e['kind'] in ('replan', 'water'):
            L.append(f'- t={e["t_sim"]:.0f} s: {e["msg"]}')
    L += ['', '## Map', '', '![map](map.png)', '', '## Timings', '',
          f'- VLM time: {metrics["vlm_s"]} s over {metrics["n_vlm_calls"]} calls '
          f'({metrics["prompt_tokens"]} prompt + {metrics["gen_tokens"]} generated tokens; '
          f'prefill {metrics["prompt_ms"] / 1000:.0f} s, decode {metrics["gen_ms"] / 1000:.0f} s)',
          f'- Flight time: {metrics["flight_sim_s"]} sim-s ({metrics["flight_wall_s"]} s wall)',
          f'- Total: {metrics["total_wall_s"]} s wall',
          f'- Peak RSS of the stack: {metrics["ram_peak_mb"].get("total")} MB', '']
    if gt:
        keys = ['precision', 'recall', 'loc_err_survey_m', 'loc_err_final_m', 'posture_acc', 'location_acc',
                'kendall_tau', 'critical_in_topk', 'level_acc', 'safe_beacon_rate', 'replan_correct']
        L += ['## Metrics (vs ground truth)', '', '| metric | value |', '|---|---|']
        for k in keys:
            v = metrics.get(k)
            L.append(f'| {k} | {v:.2f} |' if isinstance(v, float) else f'| {k} | {v} |')
    L += ['', '## Event log', '']
    L += [f'- `{e["t_sim"]:6.1f}` **{e["kind"]}** {e["msg"]}' for e in m.events]
    with open(os.path.join(rd, 'report.md'), 'w') as f:
        f.write('\n'.join(L) + '\n')
