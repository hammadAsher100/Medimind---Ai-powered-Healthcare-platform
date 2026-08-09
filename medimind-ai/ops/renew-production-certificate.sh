#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/opt/medimind/app/medimind-ai"
CERTBOT_ROOT="/opt/medimind/certbot"
CERTBOT_CONF="$CERTBOT_ROOT/conf"
CERTBOT_WEBROOT="$CERTBOT_ROOT/www"
CERTBOT_WORK="$CERTBOT_ROOT/work"
CERTBOT_LOGS="$CERTBOT_ROOT/logs"

test -f "$APP_DIR/docker-compose.yml"
test -d "$CERTBOT_CONF"
mkdir -p \
  "$CERTBOT_WEBROOT/.well-known/acme-challenge" \
  "$CERTBOT_WORK" \
  "$CERTBOT_LOGS"

# Command-line webroot options override an older standalone authenticator in
# the renewal file, so Nginx can keep serving port 80 during renewal.
docker run --rm \
  -v "$CERTBOT_CONF:/etc/letsencrypt" \
  -v "$CERTBOT_WEBROOT:/var/www/certbot" \
  -v "$CERTBOT_WORK:/var/lib/letsencrypt" \
  -v "$CERTBOT_LOGS:/var/log/letsencrypt" \
  certbot/certbot:latest renew \
  --webroot \
  --webroot-path /var/www/certbot

cd "$APP_DIR"
docker compose exec -T nginx nginx -t
docker compose exec -T nginx nginx -s reload
