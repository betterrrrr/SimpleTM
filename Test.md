cd /data/hejianbiao2023/my_docker_workspace/创新实践2/SampleTM

docker run --rm -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -e NVIDIA_VISIBLE_DEVICES=0 -e CUDA_VISIBLE_DEVICES=3 \
  -v "$PWD":/workspace -w /workspace \
  simpletm:latest \
  bash run_all.sh