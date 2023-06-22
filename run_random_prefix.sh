#WORKDIR="/data/pretrain-attention/TransCoder"
WORKDIR="/cn/projects/pe_code/TransCoder"
#HUGGINGFACE_LOCALS="/data/huggingface_models/"
HUGGINGFACE_LOCALS="/cn/pre-trained-lm/"
#ORIGIN_MODEL_DIR="/data/huggingface_models/"
ORIGIN_MODEL_DIR="/cn/projects/code/CodePrompt/save_models"
export PYTHONPATH=$WORKDIR

MODEL_NAME=${1}
#codebert
META_TASK=${2}
#translate2cls
SUB_TASK=${3:-none}
#python

DATA_NUM=-1
MODEL_DIR=save_models
SUMMARY_DIR=tensorboard
FULL_MODEL_TAG=${MODEL_NAME}

if [[ ${SUB_TASK} == none ]]; then
  OUTPUT_DIR=${MODEL_DIR}
  RES_DIR=results/${META_TASK}/${FULL_MODEL_TAG}
  RES_FN=results/${META_TASK}/${FULL_MODEL_TAG}.txt
else
  OUTPUT_DIR=${MODEL_DIR}
  RES_DIR=results/${META_TASK}/${SUB_TASK}/${FULL_MODEL_TAG}
  RES_FN=results/${META_TASK}/${SUB_TASK}/${FULL_MODEL_TAG}.txt
fi

CACHE_DIR=${WORKDIR}/.cache/${META_TASK}/${SUB_TASK}/${FULL_MODEL_TAG}
LOG=${OUTPUT_DIR}/train.log
mkdir -p ${OUTPUT_DIR}
mkdir -p ${CACHE_DIR}
mkdir -p ${RES_DIR}

RUN_FN=${WORKDIR}/learner.py

if [[ ${SUB_TASK} == none ]]; then
  # CUDA_VISIBLE_DEVICES=0 \
  TOKENIZERS_PARALLELISM=false \
    python ${RUN_FN} ${MULTI_TASK_AUG} --save_last_checkpoints --always_save_model \
    --do_meta_train 0 --prefix_type 'random' --work_dir ${WORKDIR} --meta_task ${META_TASK} --model_name ${MODEL_NAME} --data_num ${DATA_NUM}  --origin_model_dir ${ORIGIN_MODEL_DIR} \
    --output_dir ${OUTPUT_DIR}  --summary_dir ${SUMMARY_DIR} --huggingface_locals ${HUGGINGFACE_LOCALS}\
    --data_dir ${WORKDIR}/data  --prefix_tuning 'prefix_tuning' --knowledge_usage 'separate' --old_prefix_dir ${WORKDIR}/data_prefix --cache_path ${CACHE_DIR} --res_dir ${RES_DIR} --res_fn ${RES_FN} \
    2>&1 | tee ${LOG}
else
  # CUDA_VISIBLE_DEVICES=0 \
  TOKENIZERS_PARALLELISM=false \
    python ${RUN_FN} ${MULTI_TASK_AUG} --save_last_checkpoints --always_save_model \
    --do_meta_train 0 --prefix_type 'random' --work_dir ${WORKDIR} --meta_task ${META_TASK} --model_name ${MODEL_NAME} --data_num ${DATA_NUM}  --origin_model_dir ${ORIGIN_MODEL_DIR} \
    --output_dir ${OUTPUT_DIR}  --summary_dir ${SUMMARY_DIR} --huggingface_locals ${HUGGINGFACE_LOCALS}\
    --data_dir ${WORKDIR}/data  --prefix_tuning 'prefix_tuning' --knowledge_usage 'separate' --old_prefix_dir ${WORKDIR}/data_prefix --cache_path ${CACHE_DIR} --res_dir ${RES_DIR} --res_fn ${RES_FN} \
    2>&1 | tee ${LOG}
fi
