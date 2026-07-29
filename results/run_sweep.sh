#!/bin/bash
URL=${1:-http://localhost:8000}
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
PROMPT_TOKENS=${2:-512}
OUTPUT_TOKENS=${3:-256}
OUT=sweep.csv
rm -f "$OUT"
echo ">> Sweeping concurrency with prompt=${PROMPT_TOKENS} output=${OUTPUT_TOKENS}"
for C in 1 2 4 8 16 32 64; do
  python3 loadgen.py --url "$URL" --model "$MODEL" --concurrency "$C" --requests $((C*4)) \
    --prompt-tokens "$PROMPT_TOKENS" --output-tokens "$OUTPUT_TOKENS" --csv "$OUT"
  echo ""
done
echo ">> Sweep complete. Baseline results in $OUT"
