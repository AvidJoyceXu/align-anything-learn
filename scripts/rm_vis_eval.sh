MODEL_NAME_OR_PATH="../output/rm/slice_end" # model path

EVAL_DATASETS="../../datasets/PKU-SafeRLHF-single-dimension" # dataset path
EVAL_TEMPLATE="PKUSafeRLHF" # dataset template
EVAL_SPLIT="test" # split the dataset

OUTPUT_DIR="../output/rm-eval" # output dir

# For wandb online logging
export WANDB_API_KEY="4dd2f46439865db4e3547d39c268ff46468b8ef4"

# Source the setup script
source ./setup.sh

python ./rm_vis_eval.py \
 --model_name_or_path ${MODEL_NAME_OR_PATH} \
 --eval_datasets ${EVAL_DATASET} \
 --eval_template ${EVAL_TEMPLATE} \
 --eval_split ${EVAL_SPLIT} \
 --output_dir ${OUTPUT_DIR}

