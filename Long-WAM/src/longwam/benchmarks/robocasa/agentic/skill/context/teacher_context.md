# Task and predecessor knowledge (RoboCasa-365)

The goal is one recorded RoboCasa-365 rollout of the task requested at launch, controlled by a
persistent Codex agent and the Long-WAM student policy (a video-action world model trained
on the RoboCasa-365 Human300 demonstrations). Derive the exact objective from the native task
instruction and the visible observations, not from a hard-coded task script. This rollout does not
train the student.

The control method is copied from the successful RoboDojo/RoboLab hybrid pipeline: fresh student
inference, outcome/intent review, optional short EEF correction, then fresh observation. The gate
text is preserved verbatim. Full normal Codex tools, writable notes and persistent episode context
are retained.

Known student characteristics (formal 50-task evaluation, 50 episodes each): the student succeeds on
about two thirds of the single-step (atomic) kitchen tasks but only on roughly one in six composite
tasks and fewer unseen composite tasks. Typical composite failures are pursuing the wrong subtask
for the current phase, skipping a prerequisite (for example placing before opening), stalling after
a partially completed subtask, or repeating a completed one. Your review therefore matters most at
subtask boundaries.

Inspect images, recorded state, per-step proposal EEF goals and actual execution history. A
gripper command does not prove grasp, and a commanded target does not prove arrival. Keep uncertain
results uncertain; retract completed subgoals if later visual evidence contradicts them. Use all three
camera views, calculation, crops and notes when helpful.

See `eef_control.md` for the PandaOmron control contract. `workspace.json` identifies the Python
interpreter and artifact paths. `observations/NNN/` contains three RGB PNGs, observation JSON/NPZ;
`proposals/NNN/actions.npz` stores the 32x12 student proposal. `execution_NNN.npz`, `edit_NNN.json`,
requests, responses and `history.json` bind decisions to real control steps. These files are
read-only; use `NOTES.md` and `scratch/` for your own work.

All public reasoning summaries, assessment text, progress, working notes and final reports must be
in English. Private chain-of-thought is never requested or exported.
