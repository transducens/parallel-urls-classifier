#!/bin/bash

SEGALIGN_GLOB="$1"
TRG_LANG="$2"
BICLEANER_AI_MODEL_PATH="$3"

# Expected format of the segalign files: ... <tab> ... <tab> trg_text <tab> src_text (en) [<tab> ...]

if [[ -z "$SEGALIGN_GLOB" ]] || [[ -z "$TRG_LANG" ]] || [[ -z "$BICLEANER_AI_MODEL_PATH" ]]; then
  >&2 echo "Syntax: $(basename $0) <segalign_files_glob> <trg_lang> <bicleaner_ai_model_path>"

  exit 1
fi
if [[ ! -d "$BICLEANER_AI_MODEL_PATH" ]]; then
  >&2 echo "ERROR: bicleaner model path is not a directory"

  exit 1
fi

SUFFIX_BICLEANER=".bicleaner_scores_after_bifixer"

if [[ "$SEGALIGN_GLOB" == "-" ]]; then
  EVAL_CMD="cat"
else
  EVAL_CMD="ls $SEGALIGN_GLOB"
fi

eval "$EVAL_CMD" \
  | xargs -I{} -P1 bash -c \
    ' \
      f="{}'$SUFFIX_BICLEANER'.gz"; \
      if [[ -f "$f" ]]; then \
        echo "File already exists: $f"; \
      else \
        zcat {} | cut -f3,4 | bifixer -q --scol 1 --tcol 2 --ignore_empty --ignore_duplicates --ignore_segmentation - - "'$TRG_LANG'" en \
          | cache -k 1,2 bicleaner-ai-classify --scol 2 --tcol 1 --score_only --disable_minimal_length -q - - "'$BICLEANER_AI_MODEL_PATH'" \
          | pigz -c > "$f"; \
      fi; \
      nolines1=$(zcat "{}" | wc -l); \
      nolines2=$(zcat "$f" | wc -l); \
      if [[ "$nolines1" != "$nolines2" ]]; then echo "$nolines1 vs $nolines2 : {}"; fi \
    '
