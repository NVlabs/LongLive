# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prompts of the robot memory agent and fixed texts of the generator prompt.

The agent prompts are the ones behind the paper's robot results; change them only together with a new evaluation.
Placeholders are filled with ``str.format`` (literal braces are doubled). Line breaks inside the prompts are part of
the text, so some lines are longer than the code style allows.
"""
# ruff: noqa: E501

# ---------------------------------------------------------------- observer (both policies)
# Images 1-3 are early / middle / end frames of the segment that was just rendered.
OBSERVE = """You are the memory agent of a robot video generator. The generator renders a long, continuous robot video
segment by segment, but it only sees the latest frame, so you keep the memory.
Camera: {camera}. Robot: {robot}. Setting: {setting}.
The instruction this segment rendered: "{instr}"
Object names used so far (reuse them for the same objects): {names}
Images 1-3 are frames from early, middle and the end of the segment. Describe ONLY what you see in these images; do not
assume the instruction was carried out.
Return JSON only:
{{"view": "what the END frame shows, at most 30 words: the place and the objects with their positions",
 "objects": [{{"name": "short object name", "visible": true/false (in the END frame),
              "where": "where it is at the end, e.g. 'on the cabinet top, left of the charger', 'in the right hand', 'inside the open top drawer'",
              "look": "colour, material, shape in at most 8 words"}}],
 "containers": [{{"name": "drawer / box / fridge / lid / bag / cover ...", "state": "open" | "closed" | "not visible",
                 "contents": "what can be seen inside at the end if open, else ''"}}],
 "keep_frame": 1 | 2 | 3 (the frame that best shows this place as it is at the end; usually 3),
 "inside_view": null or {{"frame": 1 | 2 | 3, "caption": "at most 20 words"}} (a frame where an OPEN container's contents are clearly visible, if any),
 "crops": [{{"name": object name, "frame": 1 | 2 | 3, "bbox": [x0, y0, x1, y1] in 0-1000 normalized image coordinates}}]}}
List the objects the names list tracks (visible or not) plus new movable objects that appear. "crops": at most 2,
for objects the instruction handles, only if clearly visible and not in a gripper."""

# Appended to OBSERVE when the observer also judges whether the end frame shows the starting workspace (hybrid2); the
# input image is then passed as image 4.
OBSERVE_HOME = ('\nImage 4 is the input image the video started from. Also return "home_view": true if the END frame '
                '(image 3) shows the same workspace as image 4 from roughly the same viewpoint (objects may have moved), '
                'else false.')

# Instruction text given to the observer for the user's input image (observed before the video starts).
INPUT_IMAGE_INSTRUCTION = "(the user's input image, before the video starts)"

# ---------------------------------------------------------------- router and planner (v4; also run inside hybrid2)
ROUTE = """A robot video is generated segment by segment from a user's instructions. Instructions so far (oldest first):
{hist}
New instruction: "{instr}"
Question: during the NEW segment, does something come back into view that was out of view just before it? That is the
case when the camera turns back toward a place it looked at earlier (after turning away from it), when a closed drawer,
door, lid, box or bag is opened again, when a cover (cloth, towel, lid, book, upside-down vase or box) placed over
something is lifted off, or when an object hidden earlier is taken out.
It is NOT the case when the camera keeps looking at the same place, or turns to a place for the first time, or a
container is opened for the first time.
Return JSON only: {{"returns": true or false, "what": "the place or object that comes back, or ''"}}"""

SELECT = """You are the memory agent of a robot video generator. The generator renders a long, continuous robot video
segment by segment and only sees the latest frame. In the next segment something comes back into view: {what}.
Camera: {camera}. Robot: {robot}. Setting: {setting}.
Instructions so far (oldest first; the latest one is being rendered now and is not observed yet): {hist}
Ledger (what you observed): {ledger}
Memory pictures (the images above, in this order, oldest first): {bank}
New instruction for the next 8-second segment: "{instr}"
1. shows: list the ids of ALL memory pictures in which the returning place, the inside of the reopened container, or
   the uncovered object is clearly visible.
2. prompt: rewrite the instruction for the video model. Keep the user's action and camera wording. Replace vague
   references ("what you put inside", "the item from before", "it") by the object's name and look, and say what will be
   seen when it comes back, as it was the LAST time it was seen (objects, their look and positions, from the ledger and
   the newest pictures). At most 70 words, present tense.
Return JSON only: {{"shows": [memory picture ids], "prompt": "..."}}"""

# Fallback text for SELECT's {what} when the router names nothing.
SELECT_WHAT_DEFAULT = "a place or object seen before"

# hybrid2's selector: the rewrite describes only what the camera will see at the end of the instruction, so closed
# containers stay closed and hidden contents are not mentioned.
SELECT_STRICT = """You are the memory agent of a robot video generator. The generator renders a long, continuous robot video
segment by segment and only sees the latest frame. In the next segment something comes back into view: {what}.
Camera: {camera}. Robot: {robot}. Setting: {setting}.
Instructions so far (oldest first; the latest one is being rendered now and is not observed yet): {hist}
Ledger (what you observed): {ledger}
Memory pictures (the images above, in this order, oldest first): {bank}
New instruction for the next 8-second segment: "{instr}"
1. shows: list the ids of ALL memory pictures in which the returning place, the inside of the reopened container, or
   the uncovered object is clearly visible.
2. prompt: rewrite the instruction for the video model. Keep the user's action and camera wording. Replace vague
   references ("what you put inside", "the item from before", "it") by the object's name and look, and say what will be
   seen when it comes back, as it was the LAST time it was seen (objects, their look and positions, from the ledger and
   the newest pictures). Describe only what the camera will actually see at the end of this instruction: a drawer, lid,
   door, box or bag stays closed unless this instruction opens it, and things inside a closed container or under a
   cover are not mentioned. Whether a container is open or closed follows the latest instructions (what was closed last
   is still closed), not older pictures. At most 70 words, present tense.
Return JSON only: {{"shows": [memory picture ids], "prompt": "..."}}"""

REWRITE = """You are the memory agent of a robot video generator that renders a long, continuous robot video segment by
segment. Rewrite the user's next instruction for the video model.
Camera: {camera}. Robot: {robot}. Setting: {setting}.
Instructions so far (oldest first): {hist}
Ledger (what you observed): {ledger}
New instruction: "{instr}"
Keep the user's action and camera wording. Replace vague references ("it", "the item from before") by the object's
name and look from the ledger; name the objects the robot handles with their look. Do not describe places that are out
of view. At most 60 words, present tense.
Return JSON only: {{"prompt": "..."}}"""

# ---------------------------------------------------------------- hide events (hybrid2)
# Image 1 is the frame right before the segment, images 2-13 are 12 frames of the segment.
HIDE = """You keep the visual memory of a robot video generator. Camera: {camera}. Robot: {robot}. Setting: {setting}.
Image 1 is the frame right before the last 8-second segment. Images 2-{n} are frames of that segment, in time order.
The instruction the segment rendered: "{instr}"
Does this segment HIDE something that may be seen again later? That is the case when a drawer, door, lid, box, bag or
book is closed, when a cover (cloth, towel, lid, upside-down vase or box, book) is put over something, when an object is
put inside something that is then closed, or when the camera turns, tilts or pans away from the area it was looking at.
Count something ONLY if it is clearly visible in an earlier image AND out of view in the LAST image (shut inside, covered,
or outside the frame). Opening something, taking something out, picking something up or moving it while it stays in
view does NOT hide anything.
For each hidden thing (at most 2) return:
 what  = what becomes hidden, concretely and with its state, e.g. "the green tube lying in the open top-left drawer" or
         "the desk with the laptop, the white charger and the closed drawers"; call containers and objects by the words
         the instruction uses,
 kind  = "inside" (the contents of a container, or what is under a cover) or "area" (an area the camera turns away from),
 frame = the number of the LAST image in which that thing is still clearly and fully visible: for "inside", the open
         container with its contents in view, before it is closed or covered; for "area", the area seen head-on, before
         the camera starts to turn away. Use 1 if it is already hidden or turning away in image 2.
Describe only what you see. Return JSON only: {{"hides": [{{"what": "...", "kind": "inside", "frame": 5}}]}}
(an empty list if nothing is hidden in this segment)."""

ROUTE6 = """A robot video is generated segment by segment from a user's instructions. Instructions so far (oldest first):
{hist}
New instruction: "{instr}"
Things that were hidden earlier (id [type]: segment in which it was hidden - what was hidden):
{events}
Question: during the NEW segment, does one of these hidden things come back into view? That is the case when the camera
turns back toward an area it turned away from, when a closed drawer, door, lid, box or bag is opened again, when a cover
is lifted off, or when a hidden object is taken out. It is NOT the case when the camera keeps looking at the same place,
turns to a place for the first time, something is opened for the first time, or the new instruction itself closes or
covers something.
First decide HOW it comes back:
  "camera" = the camera turns, pans or tilts back toward a place. Only an [area] entry can come back this way, even if
             that place contains a closed drawer or box: a closed container does not show its contents.
  "open"   = a closed drawer, door, lid, box or bag is opened again, a cover is lifted, or a hidden object is taken out
             or reached for inside a container. Only a [contents] entry can come back this way.
If one comes back, give its id (if the same thing was hidden several times, the newest). H0 is the starting view of the
video: use it only for a camera return to the starting area when no other [area] entry matches.
Return JSON only: {{"returns": true or false, "how": "camera" or "open" or "", "id": "H<k>" or "", "what": "what comes back, or ''"}}"""

# The hide event H0 that stands for the input image.
START_VIEW_EVENT = "the starting view of the video (the input image)"

# hybrid2, camera return to an area: the user's instruction verbatim + "<What was hidden>" + this sentence.
CAMERA_RETURN_SUFFIX = " comes back into view exactly as it looked before the camera turned away."

# ---------------------------------------------------------------- generator prompt
SOUND = "Quiet room tone with faint robot motor whirs."
