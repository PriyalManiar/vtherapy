#!/bin/bash
# Strip audio from sample exercise videos (creates silent copies)

set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

for dir in "static/videos" "sample_exercises"; do
  if [ -d "$dir" ]; then
    for f in "$dir"/*.mp4; do
      [ -f "$f" ] || continue
      tmp="${f}.tmp.mp4"
      echo "Stripping audio from $f..."
      ffmpeg -y -i "$f" -an -c:v copy "$tmp" 2>/dev/null
      mv "$tmp" "$f"
      echo "  Done."
    done
  fi
done
echo "All sample videos are now silent."
