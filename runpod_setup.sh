#!/bin/bash
#run this once on the pod, from the folder with main_functionized3.py and stories.txt in it:
#   bash runpod_setup.sh
set -e

#which gpu we got, and its cuda driver version
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

#cupy = numpy on the gpu. cupy-cuda12x is for cuda 12 (what runpod's pytorch templates have).
#if nvidia-smi says "CUDA Version: 11.x" at the top, use cupy-cuda11x instead
pip install --quiet numpy tiktoken cupy-cuda12x

#check cupy can actually see the gpu and do math on it
python -c "import cupy; a = cupy.ones((1000, 1000)); print('cupy works, gpus:', cupy.cuda.runtime.getDeviceCount(), ' test sum:', float((a @ a).sum()))"

mkdir -p runs
echo
echo "ready. train with:"
echo "   nohup python main_functionized3.py > runs/pod1.log 2>&1 &"
echo "watch it with:"
echo "   tail -f runs/pod1.log"
