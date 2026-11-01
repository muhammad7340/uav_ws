"""Validate a VLM-generated mission plan (FLOOD_VLM_PLAN.md 3.1)."""
from __future__ import annotations

from flood_vlm.schemas import TOOLS

ALT_MIN, ALT_MAX = 5.0, 25.0
SURVEY_ALT_MIN = 18.0   # the survey grid only covers the area from >= 18 m (NOTES.md)


def survey_cells(cfg: dict) -> list[list[float]]:
    """Image centres that cover the area at survey_alt.

    The down camera's short image side (25 m at 20 m, ~21 m at roof height) lies
    along world x when the drone faces +x, the long side (34 m / ~29 m) along y,
    so the 60 x 60 m area needs a 3 x 2 grid (the plan's 4 cells leave gaps).
    """
    a = cfg['area']
    cx, cy = (a['xmin'] + a['xmax']) / 2, (a['ymin'] + a['ymax']) / 2
    # pulled ~1/3 of a footprint towards the centre so neighbouring frames overlap ~2-3 m
    dx = (a['xmax'] - a['xmin']) * 19 / 60
    dy = (a['ymax'] - a['ymin']) * 13 / 60
    xs, ys = [cx - dx, cx, cx + dx], [cy - dy, cy + dy]
    # boustrophedon order starting at the home (south) side
    return [[x, ys[0]] for x in xs] + [[x, ys[1]] for x in reversed(xs)]


def default_plan(cfg: dict) -> dict:
    alt = float(cfg['survey_alt'])
    return {'steps': [
        {'tool': 'takeoff', 'args': {'alt': alt}},
        {'tool': 'survey', 'args': {'alt': alt, 'cells': survey_cells(cfg)}},
        {'tool': 'triage', 'args': {}},
        {'tool': 'mark_beacon', 'args': {'person_id': 'all'}},
        {'tool': 'report', 'args': {}},
        {'tool': 'return_home', 'args': {}},
    ]}


def validate(plan: dict, cfg: dict) -> tuple[bool, list[str]]:
    errs: list[str] = []
    steps = plan.get('steps') if isinstance(plan, dict) else None
    if not isinstance(steps, list) or not steps:
        return False, ['plan has no steps']
    a = cfg['area']
    for i, s in enumerate(steps):
        tool = s.get('tool') if isinstance(s, dict) else None
        args = s.get('args', {}) if isinstance(s, dict) else {}
        if tool not in TOOLS:
            errs.append(f'step {i}: unknown tool {tool!r}')
            continue
        if not isinstance(args, dict):
            errs.append(f'step {i}: args must be an object')
            continue
        if tool in ('takeoff', 'survey'):
            alt = args.get('alt')
            if not isinstance(alt, (int, float)) or not ALT_MIN <= alt <= ALT_MAX:
                errs.append(f'step {i} {tool}: alt {alt!r} not in {ALT_MIN:.0f}-{ALT_MAX:.0f} m')
        if tool == 'survey':
            alt = args.get('alt')
            if isinstance(alt, (int, float)) and alt < SURVEY_ALT_MIN:
                errs.append(f'step {i} survey: alt {alt} m is too low to cover the area with these cells '
                            f'(use {SURVEY_ALT_MIN:.0f}-{ALT_MAX:.0f} m)')
            cells = args.get('cells')
            if not isinstance(cells, list) or not cells:
                errs.append(f'step {i} survey: cells missing')
            else:
                for c in cells:
                    if (not isinstance(c, list) or len(c) != 2
                            or not all(isinstance(v, (int, float)) for v in c)):
                        errs.append(f'step {i} survey: bad cell {c!r}')
                    elif not (a['xmin'] <= c[0] <= a['xmax'] and a['ymin'] <= c[1] <= a['ymax']):
                        errs.append(f'step {i} survey: cell {c} outside the area')
        if tool in ('look_again', 'mark_beacon') and not isinstance(args.get('person_id'), str):
            errs.append(f'step {i} {tool}: person_id missing')
    if steps[0].get('tool') != 'takeoff':
        errs.append('plan must start with takeoff')
    if steps[-1].get('tool') != 'return_home':
        errs.append('plan must end with return_home')
    if not any(s.get('tool') == 'survey' for s in steps if isinstance(s, dict)):
        errs.append('plan has no survey')
    return not errs, errs


def normalize(plan: dict, cfg: dict) -> dict:
    """Make a valid plan complete: ensure triage, mark_beacon and report happen before return_home.

    The planner often stops at 'survey'; the mission text always asks for triage and
    beacons, so missing steps are appended (and logged by the caller).
    """
    steps = [dict(s) for s in plan['steps']]
    tools = [s['tool'] for s in steps]
    tail = steps.pop()          # return_home
    for tool, args in (('triage', {}), ('mark_beacon', {'person_id': 'all'}), ('report', {})):
        if tool not in tools:
            steps.append({'tool': tool, 'args': args})
    steps.append(tail)
    return {'steps': steps}
