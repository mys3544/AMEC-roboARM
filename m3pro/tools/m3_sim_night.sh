#!/usr/bin/env bash
# Simulated clean-ups all night (NEXT.md, "This week"): worker W runs `m3_sim.py episode` with seeds of its
# own, one after another, until out/sim/STOP exists. `wait` holds it until the ACT run on Dionysos has ended
# (its VRAM). From m3pro/:  bash tools/m3_sim_night.sh 1 now &  bash tools/m3_sim_night.sh 2 wait &  ...
cd "$(dirname "$0")/.." || exit 1
W=$1
if [ "$2" = wait ]; then
    until ssh -o BatchMode=yes Hercules@147.232.40.154 "findstr /C:\"End of training\" C:\Users\Hercules\roboarm-lerobot\m3_act_20261026ft.log" 2>/dev/null | grep -q "End of training"; do
        sleep 120
    done
fi
seed=$((100000 * W))
while [ ! -f out/sim/STOP ]; do
    timeout 1800 ../.venv/Scripts/python.exe tools/m3_sim.py episode --seed "$seed" --pieces $((3 + seed % 3)) \
        2>&1 | grep -E "^planned|pictures, picks|stayed black|Error|SystemExit|nothing picked" | sed "s/^/w$W s$seed: /" \
        >> out/sim/night_w"$W".log
    seed=$((seed + 1))
done
echo "w$W stopped at seed $seed" >> out/sim/night_w"$W".log
