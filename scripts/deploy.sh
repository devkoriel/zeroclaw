#!/usr/bin/env bash
# Deploy ZeroClaw: build → copy to .app → sign → restart daemon
set -euo pipefail

APP="/Applications/ZeroClaw.app"
BIN="${APP}/Contents/MacOS/zeroclaw"
PLIST="$HOME/Library/LaunchAgents/com.zeroclaw.daemon.plist"
SERVICE="com.zeroclaw.daemon"
UID_VAL=$(id -u)

echo "🔨 Building release..."
cargo build --release

echo "⏹️  Stopping daemon..."
launchctl bootout "gui/${UID_VAL}" "${PLIST}" 2>/dev/null || true
sleep 1

echo "📦 Copying binary to app bundle..."
cp target/release/zeroclaw "${BIN}"

echo "🔏 Code signing..."
codesign --force --deep --sign - "${APP}"

echo "▶️  Starting daemon..."
launchctl bootstrap "gui/${UID_VAL}" "${PLIST}"
sleep 1

# Verify
if launchctl print "gui/${UID_VAL}/${SERVICE}" 2>/dev/null | grep -q "state = running"; then
    echo "✅ ZeroClaw deployed and running"
else
    echo "❌ Daemon may not be running — check: launchctl print gui/${UID_VAL}/${SERVICE}"
    exit 1
fi
