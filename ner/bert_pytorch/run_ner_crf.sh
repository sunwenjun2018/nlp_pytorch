CURRENT_DIR=`pwd`
export BERT_BASE_DIR=$CURRENT_DIR/prev_trained_model/roberta_wwm_large_ext
export GLUE_DIR=$CURRENT_DIR/CLUEdatasets
export OUTPUR_DIR=$CURRENT_DIR/outputs
TASK_NAME="cluener"

python run_ner_crf.py \
  --model_type=bert \
  --model_name_or_path=$BERT_BASE_DIR \
  --task_name=$TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir=$GLUE_DIR/${TASK_NAME}/ \
  --train_max_seq_length=128 \
  --eval_max_seq_length=512 \
  --per_gpu_train_batch_size=24 \
  --per_gpu_eval_batch_size=24 \
  --learning_rate=3e-5 \
  --num_train_epochs=5.0 \
  --logging_steps=448 \
  --save_steps=448 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --overwrite_output_dir \
  --seed=42

python run_ner_crf.py  --model_type=bert   --model_name_or_path=D:/project/python_wp/nlp/data/chinese_L-12_H-768_A-12/  --task_name=cluener  --do_train   --do_eval   --do_lower_case   --data_dir=CLUEdatasets/cluener/   --train_max_seq_length=128   --eval_max_seq_length=512   --per_gpu_train_batch_size=24   --per_gpu_eval_batch_size=24   --learning_rate=3e-5   --num_train_epochs=5.0   --logging_steps=448   --save_steps=448   --output_dir=outputs/cluener_output/   --overwrite_output_dir  --seed=42

