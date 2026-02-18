#!/bin/bash
set -e

RUN_ROOT="/Volumes/VIXinSSD/wayfinder/benchmarks/mlx/post_reboot_20260211_20260211T202821Z/decode_dense_backend_20260212T171140Z"
SIX_TASK="/Volumes/VIXinSSD/wayfinder/benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/quality_eval_glm47_consumer_v1.json"
HELDOUT="/Volumes/VIXinSSD/wayfinder/benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/quality_eval_glm47_consumer_extract_holdout_v1.json"
BENCH="python3 /Volumes/VIXinSSD/wayfinder/scripts/bench_glm_consumer_mlx.py"
COMMON="--model-path mlx-community/GLM-4.7-Flash-4bit --seq-lens 8192 --decode-len 64 --repeats 3 --chunk-size 4096 --kv-step 4096 --cooldown-sec 0 --window 64 --head-chunk-size 2 --query-chunk-size 192 --skip-multi-turn"

SEEDS="42 7 99"
MODES="dense wayfinder sparse"

total=18
count=0

for seed in $SEEDS; do
  for mode in $MODES; do
    for ds_label in six_task heldout; do
      count=$((count + 1))
      if [ "$ds_label" = "six_task" ]; then
        ds_path="$SIX_TASK"
      else
        ds_path="$HELDOUT"
      fi
      out_dir="${RUN_ROOT}/${ds_label}/s${seed}_${mode}"
      echo ""
      echo "=== [$count/$total] seed=$seed mode=$mode dataset=$ds_label ==="
      echo "    out_dir=$out_dir"
      echo ""
      $BENCH --mode $mode --seed $seed --quality-dataset "$ds_path" $COMMON --out-dir "$out_dir"
      echo "=== DONE [$count/$total] seed=$seed mode=$mode dataset=$ds_label ==="
    done
  done
done

echo ""
echo "=== ALL $total RUNS COMPLETE ==="
echo "Run root: $RUN_ROOT"
