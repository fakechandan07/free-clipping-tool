#!/bin/bash
# One-time setup for Five Star Clipper (macOS). Run:  ./setup.sh
# Installs the free tools it needs, downloads the AI models, and puts a
# "Five Star Clipper" icon on your Desktop. Safe to run again any time.
set -u
cd "$(dirname "$0")" || exit 1
DIR="$(pwd)"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

say_step() { printf '\n\033[1;35m▶ %s\033[0m\n' "$1"; }
fail() { printf '\n\033[1;31m✗ %s\033[0m\n' "$1"; exit 1; }

[ "$(uname)" = "Darwin" ] || fail "This clipper only runs on macOS."

say_step "Checking Homebrew (the free Mac package manager)"
if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew isn't installed. Install it first by pasting this into Terminal:"
  echo
  echo '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
  echo
  fail "Then run ./setup.sh again."
fi
echo "✓ Homebrew found"

say_step "Installing tools: ffmpeg (video), whisper-cpp (captions), deno (YouTube), Python"
for pkg in ffmpeg whisper-cpp deno; do
  if brew list --versions "$pkg" >/dev/null 2>&1; then echo "✓ $pkg"; else brew install "$pkg" || fail "Couldn't install $pkg"; fi
done

# the AI packages (onnxruntime) don't support the very newest Python yet: use 3.11-3.14
PY=""
for v in 3.14 3.13 3.12 3.11; do
  for c in "$(brew --prefix)/bin/python$v" "$(command -v python$v 2>/dev/null)"; do
    if [ -n "$c" ] && [ -x "$c" ]; then PY="$c"; break 2; fi
  done
done
if [ -z "$PY" ]; then
  brew install python@3.13 || fail "Couldn't install Python"
  PY="$(brew --prefix python@3.13)/bin/python3.13"
fi
echo "✓ using $("$PY" -V)"
# an environment made with an unsupported Python (e.g. 3.15) gets rebuilt
if [ -x .venv/bin/python ] && ! .venv/bin/python -c 'import sys; sys.exit(not (3, 11) <= sys.version_info[:2] <= (3, 14))'; then
  rm -rf "${DIR:?}/.venv"
fi

say_step "Setting up Python packages"
if [ ! -x .venv/bin/python ]; then
  "$PY" -m venv .venv || fail "Couldn't create the Python environment"
fi
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt || fail "Couldn't install the Python packages"
echo "✓ Python packages installed"

say_step "Downloading AI models"
./download_models.sh || echo "(You can run ./download_models.sh again later.)"

say_step "Adding a Desktop icon"
chmod +x clip open-clipper.sh download_models.sh 2>/dev/null
APP="$HOME/Desktop/Five Star Clipper.app"
script="-- Five Star Clipper: opens the clipper page (starting the clipper if needed)
do shell script \"/bin/bash \" & quoted form of \"$DIR/open-clipper.sh\" & \" > /dev/null 2>&1\""
if [ -e "$APP" ]; then
  echo "✓ $APP already exists (delete it and run setup again to point it here)"
elif osacompile -o "$APP" -e "$script" >/dev/null 2>&1; then
  echo "✓ Double-click \"Five Star Clipper\" on your Desktop to start"
else
  echo "Couldn't make the Desktop icon. Double-click \"Five Star Clipper (Mac)\" in the main folder instead."
fi

printf '\n\033[1;32m✓ All set!\033[0m Opening the clipper… Next time, just double-click "Five Star Clipper (Mac)"\n'
printf '(in this folder or on your Desktop). You can close this window.\n\n'
bash "$DIR/open-clipper.sh" >/dev/null 2>&1 &
