# use the same command as training except the script
# for example:
# bash scripts/eval_policy.sh dp3 adroit_hammer 0322 0 0
# bash scripts/eval_policy.sh dp3 maniskill_stack 0112 42 0
# bash scripts/eval_policy.sh gsplat_dp3 maniskill_gs_stack 0112 42 0 best
# bash scripts/eval_policy.sh wrist_cam_gsplat_dp3 maniskill_wrist_cam_gsworld_stack full_dataset_training 42 0 best 


DEBUG=False

alg_name=${1}
task_name=${2}
config_name=${alg_name}
addition_info=${3}
seed=${4}
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"

gpu_id=${5}
checkpoint_tag=${6}

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

python eval.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt} \
                            ${ckpt_arg}