#!/bin/bash

DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

WMT16_SCRIPT="$DIR/../parallel_urls_classifier/evaluation/wmt16.py"

if [[ ! -f "$WMT16_SCRIPT" ]]; then
  >&2 echo "ERROR: wmt16.py script not found: checked path: $WMT16_SCRIPT"

  exit 2
fi

LETT_DIR="$1" # e.g. $PWD/wmt16/lett.test
PREFIX="$2" # e.g. $PWD
GS="$3" # e.g. $PWD/test.pairs
TASKS="$4"
PUC_DIR="$5"
WMT16_FLAGS="$6"

if [[ ! -d "$LETT_DIR" ]] || ([[ ! -f "$GS" ]] && [[ ! -h "$GS" ]]) || [[ -z "$TASKS" ]] \
   || [[ -z "$PUC_DIR" ]] || ([[ "$PUC_DIR" != "src-trg" ]] && [[ "$PUC_DIR" != "trg-src" ]]) \
   || [[ -z "$WMT16_FLAGS" ]] || ([[ "$WMT16_FLAGS" != "none" ]] && [[ "$WMT16_FLAGS" != "r11" ]] \
      && [[ "$WMT16_FLAGS" != "sr" ]] && [[ "$WMT16_FLAGS" != "r11-sr" ]]); then
  >&2 echo "Syntax: script <lett_dir> <path_prefix> <gold_standard> <tasks=urls|language|langid> <dir=src-trg|trg-src>" \
           "<wmt16_flags=none|r11|sr|r11-sr>"

  exit 1
fi

for task in $(echo "$TASKS"); do
  if [[ $task != "urls" ]] && [[ $task != "language" ]] && [[ $task != "langid" ]]; then
    >&2 echo "Unknown task: $task"

    exit 1
  fi
done

echo "PREFIX: $PREFIX"

# Classify results
if [[ ! -d "$PREFIX" ]]; then
  echo "Classifying pairs..."
  mkdir -p "$PREFIX"

  if [[ -z "$PUC_TASKS" ]]; then
    echo "PUC: auxiliary tasks not defined"
    PUC_TASKS=""
  else
    #PUC_TASKS="--auxiliary-tasks language-identification langid-and-urls_classification"
    echo "PUC: auxiliary tasks: $PUC_TASKS"
    PUC_TASKS="--auxiliary-tasks $PUC_TASKS"
  fi

  if [[ -z "$PUC_BATCH_SIZE" ]]; then
    PUC_BATCH_SIZE="400"

    echo "PUC: batch size: $PUC_BATCH_SIZE"
  fi
  if [[ -z "$PUC_MODEL" ]]; then
    >&2 echo "Envvar PUC_MODEL is mandatory -> export PUC_MODEL=/path/to/model"

    exit 1
  fi

  #PUC_PAIRS="" # e.g. ./puc_eval/baseline/*.out

  if [[ -z "$PUC_PAIRS" ]]; then
    >&2 echo "Envvar PUC_PAIRS is mandatory -> export PUC_PAIRS='/path/to/pairs/*.common_suffix_if_necessary'"

    exit 1
  fi

  PUC_AWK_CMD=""

  if [[ -z "$PUC_PROVIDE_LANGS" ]]; then
    >&2 echo "ERROR: PUC_PROVIDE_LANGS was not defined"

    exit 1
  fi

  if [[ "$PUC_PROVIDE_LANGS" != "yes" ]] && [[ "$PUC_PROVIDE_LANGS" != "no" ]]; then
    >&2 echo "Allowed values for PUC_PROVIDE_LANGS are 'yes' or 'no'"

    exit 1
  fi

  if [[ "$PUC_DIR" == "src-trg" ]]; then
    PUC_AWK_CMD='{print $1"\t"$2'

    if [[ "$PUC_PROVIDE_LANGS" != "yes" ]]; then
        PUC_AWK_CMD=''"$PUC_AWK_CMD"'"\ten\tfr\ten\tfr"'
    fi
  elif [[ "$PUC_DIR" == "trg-src" ]]; then
    PUC_AWK_CMD='{print $2"\t"$1'

    if [[ "$PUC_PROVIDE_LANGS" != "yes" ]]; then
        PUC_AWK_CMD=''"$PUC_AWK_CMD"'"\tfr\ten\tfr\ten"'
    fi
  else
    >&2 echo "Bug? Unexpected PUC dir value: $PUC_DIR"

    exit 1
  fi

  PUC_AWK_CMD=''"$PUC_AWK_CMD"'}'

  ls $PUC_PAIRS \
    | xargs -I{} -P250 bash -c \
      'a=$(basename "$(echo {})"); \
        cat {} \
        | awk -F$'\''\t'\'' '\'' '"$PUC_AWK_CMD"' '\'' \
        | srun --gres=gpu:1 --cpus-per-task=2 --mem-per-cpu=6G \
          parallel-urls-classifier --model-input "'"$PUC_MODEL"'" --regression --inference \
            --batch-size "'"$PUC_BATCH_SIZE"'" '"$PUC_TASKS"' \
            --parallel-likelihood --inference-from-stdin \
            > "'"$PREFIX"'/${a}.out" \
            2> "'"$PREFIX"'/${a}.log"' \
            > "$PREFIX/out" \
            2> "$PREFIX/log"

  ERRORS=$(cat "$PREFIX/*.log" | egrep -a "ERROR|WARNING|Traceback" | wc -l)

  if [[ "$ERRORS" != "0" ]]; then
    >&2 echo "ERRORS were found: $ERRORS"
    exit 1
  else
    echo "Classification has finished!"
  fi
fi

# Run WMT16 eval
if [[ "$WMT16_FLAGS" == "none" ]]; then
  FLAGS_DIR_SUFFIX="no-sr_no-r11"
  FLAGS="--disable-rule-1-1 --disable-near-matchs"
elif [[ "$WMT16_FLAGS" == "r11" ]]; then
  FLAGS_DIR_SUFFIX="no-sr_r11"
  FLAGS="--disable-near-matchs"
elif [[ "$WMT16_FLAGS" == "sr" ]]; then
  FLAGS_DIR_SUFFIX="sr_no-r11"
  FLAGS="--disable-rule-1-1"
elif [[ "$WMT16_FLAGS" == "r11-sr" ]]; then
  FLAGS_DIR_SUFFIX="sr_r11"
  FLAGS=""
else
  >&2 echo "Bug? Unexpected WMT16 flags value: $WMT16_FLAGS"

  exit 1
fi

echo "Running WMT16 eval..."

AWK_CMD=""

if [[ "$PUC_DIR" == "src-trg" ]]; then
  AWK_CMD='{print $2"\t"$3"\t"$4}'
elif [[ "$PUC_DIR" == "trg-src" ]]; then
  AWK_CMD='{print $2"\t"$4"\t"$3}'
else
  >&2 echo "Bug? Unexpected PUC dir value: $PUC_DIR"

  exit 1
fi

PYTHON3_SUM_SCRIPT='import sys; print(sum([int(l.strip()) for l in sys.stdin]))'

for task in $(echo "$TASKS"); do
  OUTPUT="${PREFIX}_task-${task}_${FLAGS_DIR_SUFFIX}"

  if [[ -d "$OUTPUT" ]]; then
    >&2 echo "Directory for the task '$task' already exists: skipping: $OUTPUT"

    continue
  else
    echo "Output for task '$task': $OUTPUT"

    mkdir -p  "$OUTPUT"
  fi

  echo "Running task '$task'..."

  suffix="out.out"

  if [[ "$(ls $PREFIX/*.$suffix 2> /dev/null | wc -l)" == "0" ]]; then
    suffix="out"
  fi

  ls "$LETT_DIR"/*.lett.gz \
    | xargs -I{} -P20 bash -c \
      'a=$(basename "{}"); \
        zcat "{}" \
        | cut -f1,4,6 \
        | python3 "'"$WMT16_SCRIPT"'" - "'"$GS"'" --classifier-command ":)" \
          --classifier-results <(cat "'"$PREFIX"'/${a}.'"$suffix"'" | egrep -a ^'"$task"' | awk -F'\''\t'\'' '\'' '"$AWK_CMD"' '\'') \
          --results-are-fp '"$FLAGS"' \
          > '"$OUTPUT"'/${a}.out \
          2> '"$OUTPUT"'/${a}.log'
done

echo

# Eval
for task in $(echo "$TASKS"); do
  OUTPUT="${PREFIX}_task-${task}_${FLAGS_DIR_SUFFIX}"

  ERRORS=$(cat "$OUTPUT"/*.log | egrep -a "ERROR|Traceback" | wc -l)
  WARNINGS=$(cat "$OUTPUT"/*.log | egrep -a "WARNING" | wc -l)

  echo "Evaluation for task '$task':"

  if [[ "$WARNINGS" != "0" ]]; then
    >&2 echo "WARNINGS were found: $WARNINGS"
  fi
  if [[ "$ERRORS" != "0" ]]; then
    >&2 echo "ERRORS were found: $ERRORS"
  fi

  TP=$(cat "$OUTPUT"/*.out | fgrep -a "TN, FP, FN, TP:" | awk '{print $(NF-0)}' | python3 -c "$PYTHON3_SUM_SCRIPT")
  FN=$(cat "$OUTPUT"/*.out | fgrep -a "TN, FP, FN, TP:" | awk '{print $(NF-1)}' | python3 -c "$PYTHON3_SUM_SCRIPT")
  FP=$(cat "$OUTPUT"/*.out | fgrep -a "TN, FP, FN, TP:" | awk '{print $(NF-2)}' | python3 -c "$PYTHON3_SUM_SCRIPT")
  TN=$(cat "$OUTPUT"/*.out | fgrep -a "TN, FP, FN, TP:" | awk '{print $(NF-3)}' | python3 -c "$PYTHON3_SUM_SCRIPT")
  RECALL_DIV=$((TP + FN))

  if [[ "$RECALL_DIV" == "0" ]]; then
      RECALL="undefined_because_recall_div_is_zero"
  else
      RECALL=$(echo "scale=2; 100 * $TP/$RECALL_DIV" | bc)
  fi

  echo "TN, FP, FN, TP: $TN $FP $FN $TP"
  echo "Recall: $RECALL %"
  echo
done

echo "Done!"
