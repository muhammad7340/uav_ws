"""Ground-truth urgency and beacon rules (FLOOD_VLM_PLAN.md 1.2)."""
from flood_vlm import gt as G
from flood_vlm.config import load_config
from flood_vlm.scene_gen import SceneGen, rect


def _scene():
    return {
        'houses': [
            {'id': 'H1', 'x': 0, 'y': 0, 'roof_z': 3.0},
            {'id': 'H2', 'x': 20, 'y': 0, 'roof_z': 6.0},
        ],
        'people': [
            {'id': 'P1', 'x': 0, 'y': 0, 'z': 3.0, 'support': 'roof', 'house': 'H1', 'posture': 'sitting'},
            {'id': 'P2', 'x': 20, 'y': 0, 'z': 6.0, 'support': 'roof', 'house': 'H2', 'posture': 'standing'},
            {'id': 'P3', 'x': 10, 'y': 5, 'z': 0.7, 'support': 'water', 'house': None, 'posture': 'standing'},
            {'id': 'P4', 'x': 20, 'y': 1, 'z': 6.0, 'support': 'roof', 'house': 'H2', 'posture': 'lying'},
        ],
        'dry_zones': [
            {'id': 'D_H1', 'type': 'roof', 'house': 'H1', 'polygon': rect(0, 0, 8, 8), 'top_z': 3.0},
            {'id': 'D_H2', 'type': 'roof', 'house': 'H2', 'polygon': rect(20, 0, 8, 8), 'top_z': 6.0},
            {'id': 'D_M1', 'type': 'mound', 'polygon': rect(10, 12, 6, 6), 'top_z': 2.8},
        ],
    }


def test_urgency_levels():
    g = _scene()
    p1, p2, p3, p4 = g['people']
    assert G.urgency(g, p3, 0.8) == 'critical'          # in water
    assert G.urgency(g, p4, 0.8) == 'critical'          # lying
    assert G.urgency(g, p1, 0.8) == 'medium'            # roof 2.2 m above water
    assert G.urgency(g, p1, 2.4) == 'high'              # roof 0.6 m above water
    assert G.urgency(g, p2, 2.4) == 'medium'


def test_ranking_reorders_when_water_rises():
    g = _scene()
    low = [r['id'] for r in G.ranking(g, 0.8)]
    high = [r['id'] for r in G.ranking(g, 2.4)]
    assert low[:2] == ['P3', 'P4'] or low[:2] == ['P4', 'P3']
    assert set(low[:2]) == {'P3', 'P4'}
    # at 2.4 m P1 (0.6 m margin) becomes "high" and ranks before P2 (medium)
    assert high.index('P1') < high.index('P2')
    assert [r['level'] for r in G.ranking(g, 2.4)][2] == 'high'


def test_valid_beacon():
    g = _scene()
    p1, p2, p3, _ = g['people']
    assert G.valid_beacon(g, p1, 2.0, 0.0, 0.8)[0]          # same roof, 2 m away
    assert not G.valid_beacon(g, p1, 0.5, 0.0, 0.8)[0]      # too close
    assert not G.valid_beacon(g, p1, 5.0, 5.0, 0.8)[0]      # in water
    assert not G.valid_beacon(g, p2, 2.0, 0.0, 0.8)[0]      # another roof
    assert G.valid_beacon(g, p3, 10.0, 12.0, 0.8)[0]        # mound, 7 m
    assert not G.valid_beacon(g, p3, 10.0, 12.0, 2.7)[0]    # mound flooded


def test_point_in_polygon():
    poly = rect(0, 0, 2, 2)
    assert G.point_in_polygon(0, 0, poly)
    assert not G.point_in_polygon(1.5, 0, poly)


def test_scene_gen_invariants():
    cfg = load_config()
    for seed in range(1, 11):
        gen = SceneGen(seed, cfg)
        sdf = gen.build(overview=False)
        g = gen.gt
        assert 8 <= len(g['houses']) <= 12
        assert 6 <= len(g['people']) <= 9
        assert sdf.count('<model ') + sdf.count('<include>') + sdf.count('<actor ') <= 60
        assert any(p['posture'] == 'lying' for p in g['people'])
        assert any(p['support'] == 'water' for p in g['people'])
        # staging mound and embankments stay dry at the final water level
        final = max(cfg['water']['steps'])
        assert all(z['top_z'] > final for z in g['dry_zones'] if z['type'] in ('mound', 'road'))
        # people in water are not standing on a house or dry zone
        for p in g['people']:
            if p['support'] == 'water':
                assert G.dry_zone_at(g, p['x'], p['y']) is None
                assert not any(G.point_in_polygon(p['x'], p['y'], h['polygon']) for h in g['houses'])
            else:
                assert G.dry_zone_at(g, p['x'], p['y'])['house'] == p['house']
        # at least one person reaches "high" once the water is at its final level
        assert any(r['level'] == 'high' for r in G.ranking(g, final)), seed
