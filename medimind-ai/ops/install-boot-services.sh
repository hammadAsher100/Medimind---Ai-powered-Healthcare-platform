#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/opt/medimind/app/medimind-ai"
SYSTEMD_SOURCE="$APP_DIR/ops/systemd"

test -f "$APP_DIR/docker-compose.yml"
test -f "$SYSTEMD_SOURCE/medimind-compose.service"
test -f "$SYSTEMD_SOURCE/medimind-certbot-renew.service"
test -f "$SYSTEMD_SOURCE/medimind-certbot-renew.timer"

install -o root -g root -m 0644 "$SYSTEMD_SOURCE/medimind-compose.service" /etc/systemd/system/medimind-compose.service
install -o root -g root -m 0644 "$SYSTEMD_SOURCE/medimind-certbot-renew.service" /etc/systemd/system/medimind-certbot-renew.service
install -o root -g root -m 0644 "$SYSTEMD_SOURCE/medimind-certbot-renew.timer" /etc/systemd/system/medimind-certbot-renew.timer

systemctl daemon-reload
systemctl enable docker.service
systemctl enable --now medimind-compose.service
systemctl enable --now medimind-certbot-renew.timer

echo "MediMind boot recovery and certificate renewal services are enabled."
