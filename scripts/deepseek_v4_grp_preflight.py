#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from RL_Framework import AsyncRLConfig
from RL_Framework.infra.cost_model.preflight_planner import PreflightPlanner, synthetic_history
p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--output',required=True); a=p.parse_args()
cfg=AsyncRLConfig.from_yaml(a.config)
r=PreflightPlanner(cfg).run(synthetic_history(num_requests=32,input_len=1024,output_len=128))
if not r.applied: raise SystemExit('GRP preflight did not produce an applied plan')
Path(a.output).parent.mkdir(parents=True,exist_ok=True); r.planned_config.to_yaml(a.output); Path(a.output+'.decision.json').write_text(json.dumps(r.to_dict(),indent=2))
print(json.dumps({'train_gpus':r.planned_config.train_gpus,'rollout_gpus':r.planned_config.rollout_gpus,'output':a.output}))
