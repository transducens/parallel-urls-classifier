#!/bin/bash

DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

pushd "$DIR"

MODEL="$1"

if [[ -z "$MODEL" ]] || [[ ! -f "$MODEL" ]]; then
  >&2 echo "Syntax: absolute_path_to_model"
  exit 1
fi

srun --gres=gpu:1 --cpus-per-task=2 --mem-per-cpu=6G gunicorn --timeout 0 -w 1 --threads 10 "flask_server_wrapper:init('$MODEL')"

popd
