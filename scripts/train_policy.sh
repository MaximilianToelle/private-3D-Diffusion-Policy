# Train one experiment (3D-Diffusion-Policy/diffusion_policy_3d/config/experiment/<experiment>.yaml).
# Arguments: experiment, run label, seed, GPU id, then any Hydra overrides. Examples:
# bash scripts/train_policy.sh lift_fmb_fused_pcd hexagon_large 42 0 task.piece=hexagon_large_h100
# bash scripts/train_policy.sh lift_fmb_fused_pcd hexagon_large_sparse 42 0 task.piece=hexagon_large_h100 scene_representation=fused_pcd_sparse
# bash scripts/train_policy.sh stack_gsworld full_dataset 42 0


DEBUG=False
save_ckpt=True

experiment=${1}
addition_info=${2}
seed=${3}
gpu_id=${4}
shift 4   # the remaining arguments are Hydra overrides
exp_name=${experiment}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"


echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


if [ $DEBUG = True ]; then
    wandb_mode=offline
else
    wandb_mode=online
fi

cd 3D-Diffusion-Policy


export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}
python train.py --config-name=experiment/${experiment} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt} \
                            "$@"
