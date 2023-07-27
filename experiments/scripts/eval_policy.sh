MODELS=(
    ""
)

CKPTS=(
    ""
)

VIDEO_DIR=""

CMD="python experiments/eval_policy.py \
    --num_timesteps 100 \
    --video_save_path videos/$VIDEO_DIR \
    $(for i in "${!MODELS[@]}"; do echo "--checkpoint_path path/to/your/logs/${MODELS[$i]}/checkpoint_${CKPTS[$i]} "; done) \
    $(for i in "${!MODELS[@]}"; do echo "--wandb_run_name your-wandb-project/${MODELS[$i]} "; done) \
    --blocking \
"

echo $CMD

$CMD --goal_eep "0.3 0.0 0.1" --initial_eep "0.3 0.0 0.1"