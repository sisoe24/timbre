#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
DOCKER_BIN=${DOCKER_BIN:-docker}
HOST_ARCH=$(uname -m)


usage() {
  cat <<'EOF'
Usage:
  bash timbre-docker.sh load [image-tar]
  bash timbre-docker.sh analyze <audio-file> [output-dir] [-- extra timbre args]
  bash timbre-docker.sh batch <input-dir> [output-dir] [-- extra timbre args]
  bash timbre-docker.sh help

Commands:
  load      Import the shared Docker image tar with docker load
  analyze   Analyze one audio file with the Docker image
  batch     Analyze a folder of audio files with the Docker image

Examples:
  bash timbre-docker.sh load
  bash timbre-docker.sh analyze ~/Desktop/example.wav
  bash timbre-docker.sh batch ~/Desktop/my-sounds
  bash timbre-docker.sh analyze ~/Desktop/example.wav ~/Desktop/timbre-out -- --markdown
EOF
}


die() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}


require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}


arch_tag() {
  case "$HOST_ARCH" in
    arm64|aarch64)
      printf 'arm64\n'
      ;;
    x86_64|amd64)
      printf 'amd64\n'
      ;;
    *)
      die "unsupported CPU architecture: $HOST_ARCH"
      ;;
  esac
}


default_tar_name() {
  printf 'timbre-%s.tar\n' "$(arch_tag)"
}


default_image_name() {
  printf 'timbre:%s\n' "$(arch_tag)"
}


resolve_tar_path() {
  if [[ $# -gt 0 && -n ${1:-} ]]; then
    [[ -f "$1" ]] || die "image tar not found: $1"
    printf '%s/%s\n' "$(cd -- "$(dirname -- "$1")" && pwd -P)" "$(basename -- "$1")"
    return
  fi

  local tar_name
  tar_name=$(default_tar_name)

  local candidates=(
    "$PWD/$tar_name"
    "$PWD/dist/$tar_name"
    "$SCRIPT_DIR/$tar_name"
    "$SCRIPT_DIR/dist/$tar_name"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  done

  die "could not find $tar_name. Put it next to this script, in dist/, or pass the tar path explicitly."
}


resolve_image_name() {
  if [[ -n ${TIMBRE_IMAGE:-} ]]; then
    printf '%s\n' "$TIMBRE_IMAGE"
    return
  fi

  local arch_image
  arch_image=$(default_image_name)

  if "$DOCKER_BIN" image inspect "$arch_image" >/dev/null 2>&1; then
    printf '%s\n' "$arch_image"
    return
  fi

  if "$DOCKER_BIN" image inspect timbre >/dev/null 2>&1; then
    printf 'timbre\n'
    return
  fi

  printf '%s\n' "$arch_image"
}


absolute_path_existing() {
  [[ -e "$1" ]] || die "path not found: $1"
  printf '%s/%s\n' "$(cd -- "$(dirname -- "$1")" && pwd -P)" "$(basename -- "$1")"
}


absolute_dir() {
  mkdir -p "$1"
  cd -- "$1" && pwd -P
}


hf_cache_dir() {
  absolute_dir "${TIMBRE_HF_CACHE:-$SCRIPT_DIR/.hf-cache}"
}


ensure_image_loaded() {
  local image_name
  image_name=$(resolve_image_name)
  "$DOCKER_BIN" image inspect "$image_name" >/dev/null 2>&1 || die \
    "Docker image '$image_name' is not loaded. Run: bash timbre-docker.sh load"
  printf '%s\n' "$image_name"
}


split_extra_args() {
  EXTRA_ARGS=()
  if [[ $# -gt 0 && ${1:-} == "--" ]]; then
    shift
    EXTRA_ARGS=("$@")
  elif [[ $# -gt 0 ]]; then
    EXTRA_ARGS=("$@")
  fi
}


cmd_load() {
  local tar_path
  tar_path=$(resolve_tar_path "${1:-}")

  printf 'Loading Docker image from %s\n' "$tar_path"
  "$DOCKER_BIN" load -i "$tar_path"
}


cmd_analyze() {
  [[ $# -ge 1 ]] || die "usage: bash timbre-docker.sh analyze <audio-file> [output-dir] [-- extra timbre args]"

  local image_name audio_path output_dir_arg output_dir input_dir input_file cache_dir
  image_name=$(ensure_image_loaded)
  audio_path=$(absolute_path_existing "$1")
  input_dir=$(cd -- "$(dirname -- "$audio_path")" && pwd -P)
  input_file=$(basename -- "$audio_path")
  output_dir_arg=${PWD}/out
  if [[ $# -ge 2 && ${2:-} != "--" ]]; then
    output_dir_arg=$2
  fi
  output_dir=$(absolute_dir "$output_dir_arg")
  cache_dir=$(hf_cache_dir)

  if [[ $# -ge 2 && ${2:-} != "--" ]]; then
    shift 2
  else
    shift 1
  fi
  split_extra_args "$@"

  printf 'Running analysis with image %s\n' "$image_name"
  printf 'Input : %s\n' "$audio_path"
  printf 'Output: %s\n' "$output_dir"

  "$DOCKER_BIN" run --rm \
    -v "$input_dir:/data/in:ro" \
    -v "$output_dir:/data/out" \
    -v "$cache_dir:/root/.cache/huggingface" \
    "$image_name" analyze "/data/in/$input_file" \
    --output-dir /data/out \
    "${EXTRA_ARGS[@]}"
}


cmd_batch() {
  [[ $# -ge 1 ]] || die "usage: bash timbre-docker.sh batch <input-dir> [output-dir] [-- extra timbre args]"

  local image_name batch_dir output_dir_arg output_dir cache_dir
  image_name=$(ensure_image_loaded)
  batch_dir=$(absolute_path_existing "$1")
  [[ -d "$batch_dir" ]] || die "input path is not a directory: $batch_dir"
  output_dir_arg=${PWD}/out
  if [[ $# -ge 2 && ${2:-} != "--" ]]; then
    output_dir_arg=$2
  fi
  output_dir=$(absolute_dir "$output_dir_arg")
  cache_dir=$(hf_cache_dir)

  if [[ $# -ge 2 && ${2:-} != "--" ]]; then
    shift 2
  else
    shift 1
  fi
  split_extra_args "$@"

  printf 'Running batch analysis with image %s\n' "$image_name"
  printf 'Input : %s\n' "$batch_dir"
  printf 'Output: %s\n' "$output_dir"

  "$DOCKER_BIN" run --rm \
    -v "$batch_dir:/data/in:ro" \
    -v "$output_dir:/data/out" \
    -v "$cache_dir:/root/.cache/huggingface" \
    "$image_name" batch /data/in \
    --output-dir /data/out \
    "${EXTRA_ARGS[@]}"
}


main() {
  require_command "$DOCKER_BIN"

  local command=${1:-help}
  shift || true

  case "$command" in
    load)
      cmd_load "$@"
      ;;
    analyze)
      cmd_analyze "$@"
      ;;
    batch)
      cmd_batch "$@"
      ;;
    help|-h|--help)
      usage
      ;;
    *)
      usage
      die "unknown command: $command"
      ;;
  esac
}


main "$@"
