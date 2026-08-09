#!/bin/sh
set -eu

# Docker named volumes can retain ownership from an older image. Repair only
# directory ownership so existing uploads remain untouched and the non-root
# Django process can create new dated upload directories after every deploy.
for writable_dir in /app/media /app/staticfiles; do
    mkdir -p "$writable_dir"
    find "$writable_dir" -type d ! -user medimind -exec chown medimind:medimind {} +
done

exec gosu medimind "$@"
