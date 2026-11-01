"""Prompts (FLOOD_VLM_PLAN.md 2.2, 3.1). Short and factual; coordinates are 0-1000."""

P_SURVEY = (
    'Aerial top-down photo of a flooded town. Brown is flood water. Find every PERSON. '
    'For each give bbox_2d [x1,y1,x2,y2] in 0-1000, where they are (on_roof / in_water / '
    'on_dry_ground), posture (sitting / standing / lying / unsure), and confidence 0-1. '
    'Also give bboxes of DRY areas (dry roofs, mounds, dry raised road) and say for each road '
    'you see whether it is passable. Return JSON only.'
)

P_LOOK = (
    'Close aerial view. Is there a person near the centre (around point {point} in 0-1000)? '
    'Give bbox_2d of that person, location, posture, '
    'whether the surface they are on is dry, and the best SAFE point to place a rescue beacon: '
    'dry, flat, 2-4 m from the person (NOT on the person), reachable by boat (next to water). '
    'Give it as beacon_point_2d [x,y] in 0-1000 and one short reason in your own words.'
)

P_PLAN = """You command a rescue drone over a flooded town. Turn the responder's request into a plan.

Request: "{mission}"

Area (world metres): x {xmin}..{xmax}, y {ymin}..{ymax}. The drone starts at home ({hx}, {hy}).
Tools (use only these, in a sensible order):
- takeoff {{alt}}: climb at home to alt metres.
- survey {{alt, cells}}: photograph the area from alt metres (18-25 m); cells = list of [x, y] image centres.
  At 20 m one image covers about 25 m along x and 34 m along y, so 6 cells (3 along x, 2 along y)
  cover the area, e.g. [[-19,-13],[0,-13],[19,-13],[19,13],[0,13],[-19,13]].
- look_again {{person_id}}: fly low over a person to confirm them (the system adds these automatically).
- triage {{}}: rank everyone found by urgency.
- mark_beacon {{person_id}}: place a beacon at a safe dry spot near a person ("all" = everyone, in triage order).
- report {{}}: write the rescue report.
- return_home {{}}: fly home and land.
Rules: start with takeoff, end with return_home; altitude 5-25 m; cells inside the area.
Think step by step in "reasoning" (2-3 short sentences), then give the steps.
Return JSON {{"reasoning": "...", "steps": [{{"tool": ..., "args": {{...}}}}]}}."""

P_PLAN_FIX = 'Your plan was invalid: {errors}. Return a corrected plan as JSON.'

P_TRIAGE = """Flood rescue triage. Water level is now {water:.1f} m above ground.
People found by the drone (world metres; roof_height_est is the estimated height of what they stand on):
{people}

Rank everyone by who needs the rescue boat first. Levels: critical (in the water, or lying down /
injured), high (on a roof less than 1 m above the water), medium (safe for now).
Within a level, less height above water comes first. Give each person a short reason.
Think step by step in "reasoning" (who is in water, who is lying, whose roof is lowest), then rank.
Return JSON {{"reasoning": "...", "ranking": [{{"person_id", "level", "reason"}}]}} with every person exactly once."""

# Neutral wording on purpose: "Aerial photo of a flooded area" made the model call 27/27
# non-flooded FloodNet scenes "flooded" (NOTES.md, Phase 4).
P_VQA = (
    'Aerial drone photo. Answer the question with a single word or number. '
    'Question: {question}'
)

P_POINT = 'Aerial drone photo. {question} Answer with point_2d [x, y] in 0-1000.'
