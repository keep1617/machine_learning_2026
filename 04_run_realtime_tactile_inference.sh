#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

source /opt/ros/humble/setup.bash
set -u
export PYTHONNOUSERSITE=1

exec /home/adfa5456/anaconda3/envs/Camera/bin/python \
    "${SCRIPT_DIR}/04_realtime_tactile_inference.py" "$@"
