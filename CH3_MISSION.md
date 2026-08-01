# Chapter 3 unified S/M architecture

`env.UAVEnv` is the only production environment implementation.
`base_env.UAVEnv` is only the legacy pilot/v2/v3 constructor-signature
adapter; it inherits the production implementation and defines no `step`.

All four mission profiles (`S00`, `S10`, `S01`, `S11`) and all four
unknown-map profiles (`M00`, `M10`, `M20`, `M90`) use the same public
entries:

- runtime: `runtime.build_runtime`
- training/resume/checkpoint/evaluation: `training.train_and_evaluate`
- episode metrics: `metrics.augment_episode_metrics`
- resolved configuration: `ch3_config.build_ch3_config`
- scenario generation: `tools/build_ch3_scenarios.py`
- validation: `tools/validate_ch3.py`
- provenance audit: `tools/audit_ch3_provenance.py`
- run and bounded acceptance: `tools/run_ch3.py`

Examples:

```bash
python tools/run_ch3.py --phase generate --kind smoke --profiles all
python tools/run_ch3.py --phase validate --profiles all
python tools/run_ch3.py --phase train \
  --scenario-profile M10_MOVING_UNKNOWN_SINGLE \
  --base-candidate ch3_v3_full_reference \
  --seed 1 --episodes 3 --max-steps 20
python tools/run_ch3.py --phase acceptance --profiles all \
  --base-candidate ch3_v3_full_reference \
  --seed 1 --episodes 3 --resume-split 1 --max-steps 20 \
  --replay-size 32 --checkpoint-interval 1 \
  --evaluation-limit 1 --device cpu --restart
```

The former unknown-only generator, validator, audit, runner, runtime,
training, and metrics entries were removed after their behavior was folded
into these unified paths. Existing algorithm, reward, planner, target-motion,
28-dimensional observation, 3-dimensional action, and one-step handoff
semantics are unchanged.
