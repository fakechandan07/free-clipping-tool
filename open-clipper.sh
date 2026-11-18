#!/bin/bash
# Opens the Five Star Clipper page. If the clipper isn't running yet, starts it in
# the background first (no Terminal window needed). Used by the Five Star Clipper icon.
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR" || exit 1
URL_FILE="$DIR/clips/.dashboard_url"
LOG="$DIR/clipper.log"

is_up() {
  for f in "$DIR/clips/.dashboard_url" "$DIR/.dashboard_url"; do  # second: older versions
    if [ -f "$f" ] && curl -sf -m 2 -o /dev/null "$(cat "$f")/api/state"; then URL_FILE="$f"; return 0; fi
  done
  return 1
}

if ! is_up; then
  # keep the log from growing forever
  if [ -f "$LOG" ] && [ "$(stat -f%z "$LOG")" -gt 5000000 ]; then mv -f "$LOG" "$LOG.old"; fi
  if [ ! -x .venv/bin/python ]; then
    # first time: open Terminal and run the one-time setup (it opens the clipper when it's done)
    osascript -e 'on run argv' \
              -e 'tell application "Terminal" to do script "cd " & quoted form of (item 1 of argv) & " && bash setup.sh"' \
              -e 'tell application "Terminal" to activate' -e 'end run' "$DIR"
    exit 0
  fi
  echo "=== started $(date) ===" >> "$LOG"
  PYTHONUNBUFFERED=1 nohup .venv/bin/python clipper.py --no-open >> "$LOG" 2>&1 < /dev/null &
  for _ in $(seq 1 80); do is_up && break; sleep 0.25; done
fi

if is_up; then
  open "$(cat "$URL_FILE")"
else
  osascript -e 'display alert "Five Star Clipper couldn'"'"'t start" message "Look at clipper.log in the clipper folder to see why."'
  exit 1
fi
