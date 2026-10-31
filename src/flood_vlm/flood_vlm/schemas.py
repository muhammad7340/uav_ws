"""JSON schemas for constrained VLM output (FLOOD_VLM_PLAN.md 2.2, 3.1)."""

BBOX = {'type': 'array', 'items': {'type': 'integer', 'minimum': 0, 'maximum': 1000},
        'minItems': 4, 'maxItems': 4}
POINT = {'type': 'array', 'items': {'type': 'integer', 'minimum': 0, 'maximum': 1000},
         'minItems': 2, 'maxItems': 2}
CONF = {'type': 'number', 'minimum': 0, 'maximum': 1}
LOCATION = {'type': 'string', 'enum': ['on_roof', 'in_water', 'on_dry_ground']}
POSTURE = {'type': 'string', 'enum': ['sitting', 'standing', 'lying', 'unsure']}

SURVEY = {
    'type': 'object',
    'properties': {
        'people': {'type': 'array', 'maxItems': 12, 'items': {
            'type': 'object',
            'properties': {'bbox_2d': BBOX, 'location': LOCATION, 'posture': POSTURE,
                           'confidence': CONF},
            'required': ['bbox_2d', 'location', 'posture', 'confidence']}},
        'dry_areas': {'type': 'array', 'maxItems': 8, 'items': {
            'type': 'object',
            'properties': {'bbox_2d': BBOX,
                           'kind': {'type': 'string', 'enum': ['roof', 'mound', 'road']}},
            'required': ['bbox_2d', 'kind']}},
        'roads': {'type': 'array', 'maxItems': 4, 'items': {
            'type': 'object',
            'properties': {'bbox_2d': BBOX, 'passable': {'type': 'boolean'}},
            'required': ['bbox_2d', 'passable']}},
        'flooded_fraction': CONF,
    },
    'required': ['people', 'dry_areas', 'roads', 'flooded_fraction'],
}

LOOK = {
    'type': 'object',
    'properties': {
        'person_present': {'type': 'boolean'},
        'bbox_2d': BBOX,
        'location': LOCATION,
        'posture': POSTURE,
        'surface_dry': {'type': 'boolean'},
        'beacon_point_2d': POINT,
        'reason': {'type': 'string', 'maxLength': 160},
        'confidence': CONF,
    },
    'required': ['person_present', 'bbox_2d', 'location', 'posture', 'surface_dry',
                 'beacon_point_2d', 'reason', 'confidence'],
}

TOOLS = ['takeoff', 'survey', 'look_again', 'triage', 'mark_beacon', 'report', 'return_home']

PLAN = {
    'type': 'object',
    'properties': {
        'steps': {'type': 'array', 'minItems': 2, 'maxItems': 16, 'items': {
            'type': 'object',
            'properties': {
                'tool': {'type': 'string', 'enum': TOOLS},
                'args': {'type': 'object', 'properties': {
                    'alt': {'type': 'number'},
                    'cells': {'type': 'array', 'items': {'type': 'array', 'items': {'type': 'number'},
                                                         'minItems': 2, 'maxItems': 2}},
                    'person_id': {'type': 'string'},
                }},
            },
            'required': ['tool', 'args']}},
    },
    'required': ['steps'],
}

TRIAGE = {
    'type': 'object',
    'properties': {
        'ranking': {'type': 'array', 'items': {
            'type': 'object',
            'properties': {
                'person_id': {'type': 'string'},
                'level': {'type': 'string', 'enum': ['critical', 'high', 'medium']},
                'reason': {'type': 'string', 'maxLength': 200},
            },
            'required': ['person_id', 'level', 'reason']}},
    },
    'required': ['ranking'],
}

# FloodSim-VQA / FloodNet answers (Phase 4)
ANSWER = {
    'type': 'object',
    'properties': {'answer': {'type': 'string', 'maxLength': 40}},
    'required': ['answer'],
}
POINT_ANSWER = {
    'type': 'object',
    'properties': {'point_2d': POINT},
    'required': ['point_2d'],
}
