#!/usr/bin/env bash
# One-time setup: create a self-signed code signing certificate for ZeroClaw.
# This makes macOS TCC permissions (Screen Recording, Accessibility) persist
# across rebuilds — without it, every rebuild invalidates permissions.
set -euo pipefail

CERT_NAME="ZeroClaw Development"
KEYCHAIN="$HOME/Library/Keychains/login.keychain-db"

# Check if already exists
if security find-identity -v -p codesigning 2>/dev/null | grep -q "${CERT_NAME}"; then
    echo "Certificate '${CERT_NAME}' already exists."
    security find-identity -v -p codesigning | grep "${CERT_NAME}"
    exit 0
fi

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

cat > "$TMP/cert.cnf" << 'CNF'
[req]
distinguished_name = req_dn
x509_extensions = ext
prompt = no

[req_dn]
CN = ZeroClaw Development

[ext]
keyUsage = critical, digitalSignature
extendedKeyUsage = codeSigning
basicConstraints = critical, CA:false
CNF

echo "Generating self-signed code signing certificate..."
openssl req -x509 -newkey rsa:2048 \
    -keyout "$TMP/key.pem" -out "$TMP/cert.pem" \
    -days 3650 -nodes \
    -config "$TMP/cert.cnf" 2>/dev/null

openssl pkcs12 -export \
    -out "$TMP/cert.p12" \
    -inkey "$TMP/key.pem" -in "$TMP/cert.pem" \
    -passout pass:zeroclaw -legacy 2>/dev/null

echo "Importing into login keychain..."
security import "$TMP/cert.p12" \
    -k "$KEYCHAIN" \
    -T /usr/bin/codesign \
    -P "zeroclaw"

echo "Trusting for code signing..."
security add-trusted-cert -d -r trustRoot -p codeSign \
    -k "$KEYCHAIN" "$TMP/cert.pem"

# Verify
if security find-identity -v -p codesigning | grep -q "${CERT_NAME}"; then
    echo "Done. Certificate '${CERT_NAME}' is ready for code signing."
    echo ""
    echo "After next deploy, re-grant macOS permissions once:"
    echo "  System Settings → Privacy & Security → Screen Recording"
    echo "  Remove ZeroClaw, re-add it, toggle ON."
    echo "  Permissions will persist across future rebuilds."
else
    echo "ERROR: Certificate not found after import." >&2
    exit 1
fi
