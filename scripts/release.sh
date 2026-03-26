#!/usr/bin/env bash

set -eEuo pipefail

SOURCE_REPO_SLUG="sisoe24/timbre"
SOURCE_REPO_URL="https://github.com/${SOURCE_REPO_SLUG}"
TAP_DIR="repos/homebrew-timbre"
FORMULA_PATH="${TAP_DIR}/Formula/timbre.rb"
DEFAULT_BRANCH="main"
CHECKSUM_RETRY_COUNT=6
CHECKSUM_RETRY_DELAY=5

DRY_RUN=0
CURRENT_STEP="initialization"
MANUAL_RECOVERY_HINT="Inspect the repo state before retrying."

usage() {
  cat <<'EOF'
Usage:
  ./scripts/release.sh <patch|minor|major>
  ./scripts/release.sh --dry-run <patch|minor|major>

Publishes a Timbre release end-to-end:
  - bumps version in pyproject.toml
  - exports requirements.txt from Poetry
  - commits and pushes the source repo
  - creates and pushes the git tag
  - creates the GitHub release
  - updates the Homebrew tap formula and pushes it
EOF
}

log() {
  printf '[release] %s\n' "$*"
}

die() {
  printf '[release] ERROR: %s\n' "$*" >&2
  exit 1
}

on_error() {
  local exit_code=$1
  local line_no=$2
  printf '[release] ERROR: step failed: %s (line %s, exit %s)\n' \
    "$CURRENT_STEP" "$line_no" "$exit_code" >&2
  printf '[release] Manual recovery: %s\n' "$MANUAL_RECOVERY_HINT" >&2
  exit "$exit_code"
}

trap 'on_error $? $LINENO' ERR

print_cmd() {
  printf '+'
  local arg
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
}

run_cmd() {
  print_cmd "$@"
  if [[ $DRY_RUN -eq 0 ]]; then
    "$@"
  fi
}

ensure_repo_root_cwd() {
  local script_dir repo_root cwd
  script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
  repo_root=$(cd "${script_dir}/.." && pwd -P)
  cwd=$(pwd -P)

  [[ "$cwd" == "$repo_root" ]] || die "run this script from the source repo root: ${repo_root}"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

ensure_poetry_export_command() {
  if poetry export --help >/dev/null 2>&1; then
    return
  fi

  log "poetry export is unavailable; installing poetry-plugin-export"
  run_cmd poetry self add "poetry-plugin-export>=1.8"

  if [[ $DRY_RUN -eq 0 ]]; then
    poetry export --help >/dev/null 2>&1 || die "poetry export is still unavailable after installing poetry-plugin-export"
  fi
}

ensure_clean_repo() {
  local repo_path=$1
  local name=$2
  if [[ -n "$(git -C "$repo_path" status --porcelain)" ]]; then
    die "${name} repo has uncommitted changes: ${repo_path}"
  fi
}

ensure_branch() {
  local repo_path=$1
  local expected_branch=$2
  local name=$3
  local branch
  branch=$(git -C "$repo_path" branch --show-current)
  [[ "$branch" == "$expected_branch" ]] || die "${name} repo must be on ${expected_branch}, found ${branch:-<detached>}"
}

ensure_origin_remote() {
  local repo_path=$1
  local name=$2
  git -C "$repo_path" remote get-url origin >/dev/null 2>&1 || die "${name} repo is missing origin remote"
}

latest_semver_tag() {
  local tag
  tag=$(git tag --sort=-v:refname | awk '/^v[0-9]+\.[0-9]+\.[0-9]+$/ { print; exit }')
  [[ -n "$tag" ]] || die "no semver tag matching vX.Y.Z found"
  printf '%s\n' "$tag"
}

bump_version() {
  local latest_tag=$1
  local bump=$2

  python3 - "$latest_tag" "$bump" <<'PY'
import re
import sys

tag = sys.argv[1]
bump = sys.argv[2]
match = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)", tag)
if not match:
    raise SystemExit(f"invalid semver tag: {tag}")
major, minor, patch = map(int, match.groups())
if bump == "patch":
    patch += 1
elif bump == "minor":
    minor += 1
    patch = 0
elif bump == "major":
    major += 1
    minor = 0
    patch = 0
else:
    raise SystemExit(f"unsupported bump type: {bump}")
print(f"{major}.{minor}.{patch}")
PY
}

update_pyproject_version() {
  local new_version=$1

  if [[ $DRY_RUN -eq 1 ]]; then
    log "would update pyproject.toml version to ${new_version}"
    return
  fi

  python3 - "$new_version" <<'PY'
import pathlib
import re
import sys

new_version = sys.argv[1]
path = pathlib.Path("pyproject.toml")
text = path.read_text()
pattern = re.compile(r'(^version\s*=\s*")([^"]+)(")', re.MULTILINE)
updated, count = pattern.subn(rf'\g<1>{new_version}\g<3>', text, count=1)
if count != 1:
    raise SystemExit("could not update version in pyproject.toml")
path.write_text(updated)
PY
}

export_requirements() {
  local tmp_file
  tmp_file=$(mktemp)

  print_cmd poetry export \
    --format requirements.txt \
    --without-hashes \
    --only main \
    --output "$tmp_file"

  if [[ $DRY_RUN -eq 1 ]]; then
    rm -f "$tmp_file"
    return
  fi

  poetry export \
    --format requirements.txt \
    --without-hashes \
    --only main \
    --output "$tmp_file"

  {
    printf '# Generated from pyproject.toml via poetry export during release.\n'
    printf '# Do not edit requirements.txt by hand.\n\n'
    cat "$tmp_file"
  } > requirements.txt

  rm -f "$tmp_file"
}

update_formula() {
  local formula_path=$1
  local tarball_url=$2
  local checksum=$3

  if [[ $DRY_RUN -eq 1 ]]; then
    log "would update ${formula_path} url -> ${tarball_url}"
    log "would update ${formula_path} sha256 -> ${checksum}"
    return
  fi

  python3 - "$formula_path" "$tarball_url" "$checksum" <<'PY'
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
url = sys.argv[2]
checksum = sys.argv[3]
text = path.read_text()
text, url_count = re.subn(r'(^\s*url\s+")([^"]+)(")', rf'\g<1>{url}\g<3>', text, count=1, flags=re.MULTILINE)
text, sha_count = re.subn(r'(^\s*sha256\s+")([^"]+)(")', rf'\g<1>{checksum}\g<3>', text, count=1, flags=re.MULTILINE)
if url_count != 1 or sha_count != 1:
    raise SystemExit("could not update formula url and sha256")
path.write_text(text)
PY
}

checksum_from_release_tarball() {
  local tarball_url=$1
  local attempt checksum

  for ((attempt = 1; attempt <= CHECKSUM_RETRY_COUNT; attempt++)); do
    if checksum=$(curl -fsSL "$tarball_url" | shasum -a 256 | awk '{print $1}'); then
      [[ -n "$checksum" ]] || die "empty checksum returned for ${tarball_url}"
      printf '%s\n' "$checksum"
      return 0
    fi

    if (( attempt < CHECKSUM_RETRY_COUNT )); then
      log "tarball not ready yet; retrying in ${CHECKSUM_RETRY_DELAY}s (${attempt}/${CHECKSUM_RETRY_COUNT})"
      sleep "$CHECKSUM_RETRY_DELAY"
    fi
  done

  return 1
}

main() {
  local bump_arg latest_tag new_version version_tag tarball_url checksum
  local source_commit_msg tap_commit_msg

  if [[ $# -eq 0 ]]; then
    usage
    exit 1
  fi

  if [[ ${1:-} == "--dry-run" ]]; then
    DRY_RUN=1
    shift
  fi

  [[ $# -eq 1 ]] || {
    usage
    exit 1
  }

  bump_arg=$1
  case "$bump_arg" in
    patch|minor|major) ;;
    *)
      usage
      die "bump type must be one of: patch, minor, major"
      ;;
  esac

  ensure_repo_root_cwd

  CURRENT_STEP="preflight checks"
  MANUAL_RECOVERY_HINT="Install missing tools, ensure both repos are clean on ${DEFAULT_BRANCH}, then rerun the script."
  require_command git
  require_command gh
  require_command python3
  require_command poetry
  require_command curl
  require_command shasum
  [[ -d "$TAP_DIR" ]] || die "tap repo directory not found: ${TAP_DIR}"
  [[ -f "$FORMULA_PATH" ]] || die "formula file not found: ${FORMULA_PATH}"
  ensure_clean_repo "." "source"
  ensure_clean_repo "$TAP_DIR" "tap"
  ensure_branch "." "$DEFAULT_BRANCH" "source"
  ensure_branch "$TAP_DIR" "$DEFAULT_BRANCH" "tap"
  ensure_origin_remote "." "source"
  ensure_origin_remote "$TAP_DIR" "tap"
  run_cmd poetry check --lock
  ensure_poetry_export_command
  run_cmd gh auth status

  CURRENT_STEP="version calculation"
  MANUAL_RECOVERY_HINT="Check source repo tags and choose a valid bump type."
  latest_tag=$(latest_semver_tag)
  new_version=$(bump_version "$latest_tag" "$bump_arg")
  version_tag="v${new_version}"
  tarball_url="${SOURCE_REPO_URL}/archive/refs/tags/${version_tag}.tar.gz"
  source_commit_msg="chore(release): ${version_tag}"
  tap_commit_msg="timbre ${version_tag}"
  git rev-parse -q --verify "refs/tags/${version_tag}" >/dev/null 2>&1 && die "tag already exists: ${version_tag}"

  log "latest tag: ${latest_tag}"
  log "next version: ${version_tag}"
  log "tarball url: ${tarball_url}"

  CURRENT_STEP="update source release files"
  MANUAL_RECOVERY_HINT="Revert or fix pyproject.toml or requirements.txt if the release update is incomplete, then rerun."
  update_pyproject_version "$new_version"
  export_requirements
  run_cmd git add pyproject.toml requirements.txt

  CURRENT_STEP="commit source release"
  MANUAL_RECOVERY_HINT="Decide whether to keep or amend the source release commit before retrying."
  run_cmd git commit -m "$source_commit_msg"

  CURRENT_STEP="push source release commit"
  MANUAL_RECOVERY_HINT="Push or reconcile the source repo main branch manually before retrying."
  run_cmd git push origin "$DEFAULT_BRANCH"

  CURRENT_STEP="create source tag"
  MANUAL_RECOVERY_HINT="Delete or reuse the local tag ${version_tag} before retrying."
  run_cmd git tag -a "$version_tag" -m "$version_tag"

  CURRENT_STEP="push source tag"
  MANUAL_RECOVERY_HINT="Push or delete the remote tag ${version_tag} manually before retrying."
  run_cmd git push origin "$version_tag"

  CURRENT_STEP="create GitHub release"
  MANUAL_RECOVERY_HINT="Use gh release create ${version_tag} manually if the tag exists but the release was not created."
  run_cmd gh release create "$version_tag" --title "$version_tag" --notes ""

  CURRENT_STEP="fetch release tarball checksum"
  MANUAL_RECOVERY_HINT="Wait for the GitHub tag tarball to become available, then update the formula checksum manually if needed."
  if [[ $DRY_RUN -eq 1 ]]; then
    checksum="<dry-run>"
    log "would fetch ${tarball_url} and compute sha256 with retry"
  else
    checksum=$(checksum_from_release_tarball "$tarball_url") || die "failed to fetch checksum from ${tarball_url}"
    log "release checksum: ${checksum}"
  fi

  CURRENT_STEP="update Homebrew formula"
  MANUAL_RECOVERY_HINT="Inspect ${FORMULA_PATH} and restore or finish the formula update manually before retrying."
  update_formula "$FORMULA_PATH" "$tarball_url" "$checksum"
  run_cmd git -C "$TAP_DIR" add "Formula/timbre.rb"

  CURRENT_STEP="commit tap update"
  MANUAL_RECOVERY_HINT="Decide whether to keep or amend the tap commit before retrying."
  run_cmd git -C "$TAP_DIR" commit -m "$tap_commit_msg"

  CURRENT_STEP="push tap update"
  MANUAL_RECOVERY_HINT="Push or reconcile the tap repo main branch manually before retrying."
  run_cmd git -C "$TAP_DIR" push origin "$DEFAULT_BRANCH"

  CURRENT_STEP="complete"
  MANUAL_RECOVERY_HINT="No recovery needed."
  log "release flow complete for ${version_tag}"
}

main "$@"
