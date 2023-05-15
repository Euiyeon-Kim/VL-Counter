NUM_GPUS=1
CHECKPOINT_DIR=SAMCorrCNN/fixAll_baseline_b16_cutmux_2ce001
CONFIG_NAME=SAMCorrCNN.yaml

mkdir -p exps/${CHECKPOINT_DIR} && \
python -m torch.distributed.launch --use_env \
--nproc_per_node=${NUM_GPUS} \
train.py \
--exp_name ${CHECKPOINT_DIR} \
--config configs/${CONFIG_NAME} \
2>&1 | tee -a exps/${CHECKPOINT_DIR}/train.log
