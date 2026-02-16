#!/usr/bin/env bash
# Deploy ZeroClaw: build → copy to .app → sign → restart daemon
set -euo pipefail

# Ensure cargo is in PATH (rustup may not be in default PATH)
export PATH="$HOME/.rustup/toolchains/stable-aarch64-apple-darwin/bin:$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

APP="/Applications/ZeroClaw.app"
BIN="${APP}/Contents/MacOS/zeroclaw"
PLIST="$HOME/Library/LaunchAgents/com.zeroclaw.daemon.plist"
SERVICE="com.zeroclaw.daemon"
SIGN_IDENTITY="ZeroClaw Development"
UID_VAL=$(id -u)

echo "🔨 Building release..."
cargo build --release

echo "⏹️  Stopping daemon..."
launchctl bootout "gui/${UID_VAL}" "${PLIST}" 2>/dev/null || true
sleep 1

echo "📦 Copying binary to app bundle..."
cp target/release/zeroclaw "${BIN}"

echo "🔏 Code signing (identity: ${SIGN_IDENTITY})..."
if security find-identity -v -p codesigning | grep -q "${SIGN_IDENTITY}"; then
    codesign --force --deep --sign "${SIGN_IDENTITY}" --identifier "${SERVICE}" "${APP}"
else
    echo "⚠️  Certificate '${SIGN_IDENTITY}' not found — falling back to ad-hoc signing"
    echo "   macOS permissions (Screen Recording, Accessibility) will reset on each rebuild."
    echo "   Run: scripts/setup-cert.sh to create a persistent signing certificate."
    codesign --force --deep --sign - --identifier "${SERVICE}" "${APP}"
fi

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
