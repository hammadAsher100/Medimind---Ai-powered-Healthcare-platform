# Production certificate renewal

Production certificates are persistent host files under
`/opt/medimind/certbot/conf`. They are mounted read-only into Nginx and must
never be copied into Git or a Docker image.

Nginx serves `/.well-known/acme-challenge/` from
`/opt/medimind/certbot/www`, so renewal does not require stopping Nginx or
temporarily releasing port 80. The repository provides
`ops/renew-production-certificate.sh`, which runs Certbot with webroot options,
validates Nginx, and reloads it after renewal.

Run a dry run on EC2 before relying on automatic renewal:

```bash
sudo docker run --rm \
  -v /opt/medimind/certbot/conf:/etc/letsencrypt \
  -v /opt/medimind/certbot/www:/var/www/certbot \
  -v /opt/medimind/certbot/work:/var/lib/letsencrypt \
  -v /opt/medimind/certbot/logs:/var/log/letsencrypt \
  certbot/certbot:latest renew --dry-run \
  --webroot --webroot-path /var/www/certbot
```

If the certificate was originally created with standalone mode, the explicit
webroot options above override that authenticator while Nginx remains online.
Schedule the repository script with the host's existing systemd timer or cron
mechanism. A monthly run is sufficient because Certbot renews only certificates
that are close to expiry; it does not issue a new certificate on every run.

Example root cron entry:

```cron
17 3 1 * * /opt/medimind/app/medimind-ai/ops/renew-production-certificate.sh >> /var/log/medimind-certbot-renewal.log 2>&1
```

CI/CD verifies that the current certificate and key exist before it stops any
containers. Deployment never regenerates, deletes, or overwrites certificates.
