

docker run --gpus all --ipc=host --rm -it \
    --mount src=$(pwd),dst=/videocap,type=bind \
    --mount src=$REPO_DIR'/datasets/',dst=/videocap/datasets,type=bind$",readonly" \
    --mount src=$REPO_DIR'/models/',dst=/videocap/models,type=bind,readonly \
    --mount src=$REPO_DIR'/output/',dst=/videocap/output,type=bind \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -w /videocap linjieli222/videocap_torch1.7:mufcryan2 \
    bash -c "source /videocap/setup.sh && bash"
