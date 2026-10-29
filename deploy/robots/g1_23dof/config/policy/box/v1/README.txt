G1 23DoF autonomous research result

Plain-PD simulation; all actor inputs remain 75; no estimator or extra actor layer.
Research artifact, not hardware validated. Original physical criteria and sensor-only switching retained.

Nominal evaluation: 1024 continuous trials.
go-to: 100.00% conditional on preceding stages succeeding.
take: 99.22% conditional on preceding stages succeeding.
carry: 92.42% conditional on preceding stages succeeding.
put-down: 92.44% conditional on preceding stages succeeding.
return: 54.15% conditional on preceding stages succeeding.
Full sequence: 45.90%.

Legacy assisted-physics results are separate: an unphysical 59.9 N upward floating-base force was removed.
See report.json for checkpoint hashes, repeat evidence, failure audit and limitations.

Repeat the full evaluation:
PYTHONPATH=.deps:policies:scripts CUDA_VISIBLE_DEVICES=0 /home/rwang/humanoid/.venv/bin/python policies/five_teacher_sequence.py evaluate --bundle /home/rwang/humanoid-full/results/autoresearch-20261029/final-selection/policy.pt --out /tmp/g1-research-repeat --envs 1024 --seed 120001

Final repeat: carry 90.35%, put-down 91.83%, return 49.82%, full sequence 41.02%.

Failure audit (512 separate continuous trials of the selected actors; overlapping checks):
Carry: 44 of 47 failed handoffs missed the 8 cm box-position gate; median box error among failed handoffs was 8.94 cm.
Put-down: 38 failed trials. Among 37 sensor handoffs, 19 missed box-position, 11 instantaneous shelf-contact, and 7 root XY checks. One failed earlier.
Return: 187 of 189 failures reached a sensor handoff; all 187 missed the 10 cm root XY gate. Median first-handoff error was 17.10 cm; two failures occurred earlier.

Next: train return from exact final put-down states; strengthen endpoint accuracy and carry box-position margin; calibrate noise, delay, and PD physics on G1. Synthetic small-noise probing reduced this carry pair to 84.65%, so nominal success does not establish deployment robustness.

Verified final video: 70.90 seconds, 3,545 frames at 50 fps, one successful uninterrupted rollout. Batch completion remains 41–46%.
