@echo off
rem ACT on all of 2026-10-23's M3 recordings, the weekend's run: started detached (weekend_act.ps1), so no SSH session holds it
cd /d C:\Users\Hercules\roboarm-lerobot
.venv\Scripts\python.exe lerobot_train_win.py --dataset.repo_id=local/m3_pick_20261023b --dataset.root=C:/Users/Hercules/roboarm-lerobot/m3_pick_20261023b --dataset.eval_split=0.1 --policy.type=act --policy.device=cuda --policy.use_amp=true --policy.push_to_hub=false --policy.chunk_size=20 --policy.n_action_steps=10 --batch_size=8 --num_workers=6 --steps=100000 --log_freq=500 --eval_steps=5000 --save_freq=20000 --output_dir=C:/Users/Hercules/roboarm-lerobot/m3_act_20261023b --wandb.enable=false --job_name=m3_act_20261023b > m3_act_20261023b.log 2>&1
