#/bin/bash

LANG_="$1"
EXTRA_ARGS_="$2"
BIN_=$([[ -z "$3" ]] && echo "monofixer" || echo "$3")
PIPE_=$([[ -z "$4" ]] && echo "cat" || echo "$4")

# If you want to execute bifixer, e.g.: bash $0 "en es" "--tcol 2" "bifixer"

#$BIN_ --scol 1 --ignore_long --ignore_html --ignore_duplicates --ignore_segmentation --ignore_empty -q $EXTRA_ARGS_ - - $LANG_ | cut -f1

# Similar to Bitextor: you might want to add extra fields using $2
$BIN_ --scol 1 --ignore_duplicates -q $EXTRA_ARGS_ - - $LANG_ | eval $PIPE_
