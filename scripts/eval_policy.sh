# Evaluate a trained experiment: the same arguments as train_policy.sh plus the checkpoint tag
# (e.g. best), then any Hydra overrides. Examples:
# bash scripts/eval_policy.sh lift_fmb_fused_pcd hexagon_large 42 0 best task.piece=hexagon_large_h100
# bash scripts/eval_policy.sh stack_gsworld full_dataset_training 42 0 best


DEBUG=False

experiment=${1}
addition_info=${2}
seed=${3}
gpu_id=${4}
checkpoint_tag=${5}
shift 5   # the remaining arguments are Hydra overrides
exp_name=${experiment}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"

if [ $DEBUG = True ]; then
    wandb_mode=offline
else
    wandb_mode=online
fi

ckpt_arg=""
if [ -n "$checkpoint_tag" ]; then
    ckpt_arg="+checkpoint.checkpoint_tag='${checkpoint_tag}'"
fi

cd 3D-Diffusion-Policy

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

echo "DEBUG: Executing eval.py with ckpt_arg='${ckpt_arg}'"

python eval.py --config-name=experiment/${experiment} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            ${ckpt_arg} \
                            "$@"
