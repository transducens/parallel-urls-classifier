#!/bin/bash
#SBATCH --job-name=bicleaner_ai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH -e slurm-%j-apply_bifixer_and_bicleaner_to_segalign.err
#SBATCH -o slurm-%j-apply_bifixer_and_bicleaner_to_segalign.out

SEGALIGN_GLOB="$1"
TRG_LANG="$2"
BICLEANER_AI_MODEL_PATH="$3"
BICLEANER_EXTRA_ARGS="$4" # e.g. "-b 10000 --batch_size 64"
PROFILING="$5" # e.g. "/usr/bin/time -v"

# Expected format of the segalign files: ... <tab> ... <tab> trg_text <tab> src_text (en) [<tab> ...]

if [[ -z "$SEGALIGN_GLOB" ]] || [[ -z "$TRG_LANG" ]] || [[ -z "$BICLEANER_AI_MODEL_PATH" ]]; then
  >&2 echo -n "Syntax: $(basename $0) <segalign_files_glob> <trg_lang> <bicleaner_ai_model_path>"
  >&2 echo -n " [<bicleaner_extra_args> <profiling>]"
  >&2 echo ""

  exit 1
fi
if [[ ! -d "$BICLEANER_AI_MODEL_PATH" ]]; then
  >&2 echo "ERROR: bicleaner model path is not a directory"

  exit 1
fi

SUFFIX_BICLEANER=".bicleaner_scores_after_bifixer"
APPLY_CMD="""\
cut -f3,4 \
  | $PROFILING bifixer -q --scol 1 --tcol 2 --ignore_empty --ignore_duplicates \
    --ignore_segmentation - - "$TRG_LANG" en \
  | $PROFILING cache -k 1,2 bicleaner-ai-classify -q --scol 2 --tcol 1 --score_only \
    --disable_porn_removal --disable_minimal_length $BICLEANER_EXTRA_ARGS \
    - - "$BICLEANER_AI_MODEL_PATH" \
"""

if [[ "$SEGALIGN_GLOB" == "-" ]]; then
  OUTPUT_FILE="$PWD/$(date +%Y%m%d%H%M%S%N)$SUFFIX_BICLEANER"
  f="${OUTPUT_FILE}.gz";

  >&2 echo "Output file: $f"

  eval $APPLY_CMD \
    | pigz -c > "$f"
else
  ls $SEGALIGN_GLOB \
    | xargs -I{} -P1 bash -c \
      ' \
        f="{}'$SUFFIX_BICLEANER'.gz"; \
        if [[ -f "$f" ]]; then \
          >&2 echo "File already exists: $f"; \
        else \
          >&2 echo "Output file: $f";
          zcat {} \
            | eval '"'$APPLY_CMD'"' \
            | pigz -c > "$f"; \
        fi; \
        nolines1=$(zcat "{}" | wc -l); \
        nolines2=$(zcat "$f" | wc -l); \
        if [[ "$nolines1" != "$nolines2" ]]; then >&2 echo "$nolines1 vs $nolines2 : {}"; fi \
      '
fi
