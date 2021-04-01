export CUDA_VISIBLE_DEVICES=1

DATA_DIR=train_data
MODEL_DIR=hf_model_v1
postfix='v1'
MIN_LEN=10
MAX_LEN=82
B=32
output_pred_file=pred_test_b${B}_min${MIN_LEN}_max${MAX_LEN}_$postfix.txt
output_score_file=pred_test_b${B}_min${MIN_LEN}_max${MAX_LEN}_score_$postfix.json



python run_eval.py $MODEL_DIR \
$DATA_DIR/test.source  \
$MODEL_DIR/$output_pred_file \
--task summarization \
--score_path $MODEL_DIR/$output_score_file \
--reference_path $DATA_DIR/test.target \
--device cuda \
--fp16 \
--bs $B \
--max_length $MAX_LEN \
--min_length $MIN_LEN

files2rouge $MODEL_DIR/$output_pred_file $DATA_DIR/test.target > $MODEL_DIR/file2rouge_${output_pred_file} 2>&1