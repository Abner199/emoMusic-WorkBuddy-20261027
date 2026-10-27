#!/usr/bin/env bash
# 心音 EmoSong 本地预览：必须通过 http 访问，file:// 下摄像头与模型 fetch 都会失败
set -e
PORT="${1:-8787}"
cd "$(dirname "$0")"

echo "============================================"
echo "  心音 EmoSong · http://127.0.0.1:${PORT}/"
echo "============================================"

if command -v python3 >/dev/null 2>&1; then
  python3 -m http.server "$PORT" --bind 127.0.0.1
elif command -v python >/dev/null 2>&1; then
  python -m http.server "$PORT" --bind 127.0.0.1
else
  npx --yes serve -l "$PORT" .
fi
