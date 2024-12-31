# path to your trained reward model
# for example: ~/output/rm
MODEL_NAME_OR_PATH="../output/rm/slice_end" # model path

# for example: ~/align-anything/generate_scripts/test/Qwen-0.5B-Instruct_num_4_time_20241103_133249.json
EVAL_DATASETS="/home/pku0014/align-anything/generate_scripts/test/Qwen-0.5B-Instruct_num_4_time_20241230_040718.json" # the generated model output in json
# EVAL_DATASETS="/home/pku0014/align-anything/generate_scripts/test/slice_end_num_4_time_20241230_041550.json" # the generated model output in json

EVAL_TEMPLATE="PKUSafeRLHF" # dataset template
EVAL_SPLIT="test" # split the dataset

OUTPUT_DIR="../output/rm_score" # output dir

# For wandb online logging
export WANDB_API_KEY=""

# Source the setup script
source ./setup.sh

# Execute deepspeed command
deepspeed \
     --master_port ${MASTER_PORT} \
     --module align_anything.trainers.text_to_text.rm_score \
     --model_name_or_path ${MODEL_NAME_OR_PATH} \
     --eval_datasets ${EVAL_DATASETS} \
     --eval_template ${EVAL_TEMPLATE} \
     --eval_split ${EVAL_SPLIT} \
     --output_dir ${OUTPUT_DIR} \