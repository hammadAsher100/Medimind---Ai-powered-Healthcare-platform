#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/opt/medimind/app/medimind-ai"
CERTBOT_ROOT="/opt/medimind/certbot"
DOMAIN="app.medimind-ai.online"
EMAIL="${CERTBOT_EMAIL:?Set CERTBOT_EMAIL to the certificate notification email address}"

test -f "$APP_DIR/docker-compose.yml"
mkdir -p \
  "$CERTBOT_ROOT/conf" \
  "$CERTBOT_ROOT/www/.well-known/acme-challenge" \
  "$CERTBOT_ROOT/work" \
  "$CERTBOT_ROOT/logs"

docker run --rm \
  -v "$CERTBOT_ROOT/conf:/etc/letsencrypt" \
  -v "$CERTBOT_ROOT/www:/var/www/certbot" \
  -v "$CERTBOT_ROOT/work:/var/lib/letsencrypt" \
  -v "$CERTBOT_ROOT/logs:/var/log/letsencrypt" \
  certbot/certbot:latest certonly \
  --webroot \
  --webroot-path /var/www/certbot \
  --cert-name "$DOMAIN" \
  --domain "$DOMAIN" \
  --email "$EMAIL" \
  --agree-tos \
  --no-eff-email \
  --non-interactive

test -s "$CERTBOT_ROOT/conf/live/$DOMAIN/fullchain.pem"
test -s "$CERTBOT_ROOT/conf/live/$DOMAIN/privkey.pem"
openssl x509 -in "$CERTBOT_ROOT/conf/live/$DOMAIN/fullchain.pem" -noout -subject -issuer -enddate
