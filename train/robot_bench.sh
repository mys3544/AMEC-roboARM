#!/bin/bash
# The model table's speeds on a robot (PLAN.md M5): each ONNX in ~/roboarm/models/kit built into a
# TensorRT engine here, and timed -- GPU compute only, CUDA graphs, 10 s after 0.5 s of warm-up.
# The ONNX come from hekate (train/train_kit.py, train/train_rfdetr.py; YOLOE-26s and YOLO26n-depth
# exported there too) under these names; a missing one is skipped. Run it detached, it takes ~30 min:
#   cd ~/roboarm/models/kit && nohup ./robot_bench.sh >> bench.out 2>&1 < /dev/null &
# A plain INT8 build of yolo26n-seg fails on TensorRT 10.7 (no INT8 kernel for the mask head's proto
# conv); the QAT one builds but runs slower than FP16 (2026-10-12).
cd ~/roboarm/models/kit
T=/usr/src/tensorrt/bin/trtexec
run() {
  name=$1; onnx=$2; shift 2
  [ -f "$onnx" ] || { echo "=== $name: no $onnx"; return; }
  start=$(date +%s)
  $T --onnx=$onnx "$@" --timingCacheFile=timing.cache --useCudaGraph --noDataTransfers --warmUp=500 --duration=10 --saveEngine=$name.engine > $name.log 2>&1
  echo "=== $name: exit $?, $(( $(date +%s) - start )) s with the build"
  grep -E "GPU Compute Time:" $name.log | sed "s/^.*\] //"
}
run yolo26n_fp16 yolo26n.onnx --fp16
run yolo26n_int8 yolo26n_int8.onnx --int8 --fp16
run yolo26s_fp16 yolo26s.onnx --fp16
run yoloe26s_fp16 yoloe26s.onnx --fp16
run rfdetr_segn_fp16 rfdetr_segn.onnx --fp16
run yolo26n_depth_fp16 yolo26n_depth.onnx --fp16
echo DONE
