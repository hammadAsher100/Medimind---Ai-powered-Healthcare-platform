# MediMind On-Demand EC2 Architecture

This directory contains the infrastructure and operating-system integration that allows the existing MediMind Docker Compose application to remain stopped when unused and start automatically from an always-available web entry point.

The design intentionally keeps long-running medical report and AI requests away from CloudFront. The root domain hosts only the startup experience; the running application uses `app.medimind-ai.online` directly through Nginx.

## Resources

| Template | Region | Resources |
|---|---|---|
| `dns-template.yaml` | Any (recommended `ap-south-1`) | Retained Route 53 hosted zone |
| `template.yaml` | `ap-south-1` | Wake/status Lambda, idle Lambda, HTTP API, DynamoDB state table, schedules, least-privilege policies, optional Elastic IP |
| `edge-template.yaml` | `us-east-1` | ACM viewer certificate, private startup S3 bucket, CloudFront distribution, root/`www` aliases, `app` A record |

AWS SAM/CloudFormation is the only IaC system used. Application data, Docker volumes, certificates, models, and the existing EC2 instance are not created or replaced by these templates.

## Required parameters

- `EC2_INSTANCE_ID`
- Existing EC2 instance profile role name
- Existing GitHub OIDC deployment role name
- `APP_URL=https://app.medimind-ai.online`
- `HEALTH_CHECK_URL=https://app.medimind-ai.online/readyz`
- `IDLE_TIMEOUT_MINUTES` (default `30`)
- `MINIMUM_RUNTIME_MINUTES` (default `15`)
- A random `ORIGIN_TOKEN` of at least 32 characters

Never commit the origin token. Store it as a GitHub environment secret or pass it interactively to CloudFormation.

## Safe deployment order

### 1. Validate source before touching AWS

```bash
python -m unittest discover -s infrastructure/on-demand/tests -v
docker compose -f medimind-ai/docker-compose.yml --env-file medimind-ai/.env.example config --quiet
```

Install AWS SAM CLI, then validate both SAM and CloudFormation templates:

```bash
sam validate --lint -t infrastructure/on-demand/template.yaml
aws cloudformation validate-template --template-body file://infrastructure/on-demand/dns-template.yaml
aws cloudformation validate-template --region us-east-1 --template-body file://infrastructure/on-demand/edge-template.yaml
```

### 2. Create Route 53 hosted zone

```bash
aws cloudformation deploy \
  --stack-name medimind-dns \
  --region ap-south-1 \
  --template-file infrastructure/on-demand/dns-template.yaml \
  --parameter-overrides \
    OriginElasticIp=ELASTIC_IP \
    CutoverEntryDns=false

aws cloudformation describe-stacks \
  --stack-name medimind-dns \
  --region ap-south-1 \
  --query 'Stacks[0].Outputs'
```

This first creates root and `www` A records pointing to the existing EC2 Elastic IP. Replace the domain's Hostinger nameservers with the four `NameServers` output values only after confirming those records exist. Wait until public NS queries return Route 53 before continuing. This registrar action cannot be performed by CloudFormation.

### 3. Deploy regional control plane

```bash
sam build --template-file infrastructure/on-demand/template.yaml
sam deploy \
  --stack-name medimind-on-demand \
  --region ap-south-1 \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides \
    Ec2InstanceId=INSTANCE_ID \
    Ec2InstanceRoleName=INSTANCE_ROLE_NAME \
    GitHubDeployRoleName=MediMindGitHubDeployRole \
    AppUrl=https://app.medimind-ai.online \
    HealthCheckUrl=https://app.medimind-ai.online/readyz \
    IdleTimeoutMinutes=30 \
    MinimumRuntimeMinutes=15 \
    OriginToken=RANDOM_SECRET \
    AutomationEnabled=false
```

Record the `WakeApiOriginDomain`, `ActivityTableName`, and `OriginElasticIp` outputs.

Keep `AutomationEnabled=false` throughout migration. Enable it only after activity tracking, wake/readiness, graceful shutdown, and the public entry cutover have all passed validation.

The Django and FastAPI containers use the existing EC2 instance role to write activity leases. Require IMDSv2 and allow its response to cross the Docker network's additional hop:

```bash
aws ec2 modify-instance-metadata-options \
  --region ap-south-1 \
  --instance-id INSTANCE_ID \
  --http-endpoint enabled \
  --http-tokens required \
  --http-put-response-hop-limit 2
```

Do not place long-lived AWS keys in `.env`. The inline policy created by the regional stack restricts application access to this instance's activity-table partition.

### 4. Deploy edge stack without root-domain cutover

Deploy this stack in `us-east-1`, because CloudFront viewer certificates must exist there. This creates `app.medimind-ai.online`, the startup distribution, and its viewer certificate while the DNS stack keeps live root and `www` traffic on EC2.

```bash
aws cloudformation deploy \
  --stack-name medimind-entry \
  --region us-east-1 \
  --template-file infrastructure/on-demand/edge-template.yaml \
  --parameter-overrides \
    HostedZoneId=ROUTE53_ZONE_ID \
    ApiOriginDomain=API_GATEWAY_HOSTNAME \
    OriginElasticIp=ELASTIC_IP \
    OriginToken=RANDOM_SECRET
```

Upload the startup assets and invalidate CloudFront:

```bash
STARTUP_BUCKET=$(aws cloudformation describe-stacks --stack-name medimind-entry --region us-east-1 --query "Stacks[0].Outputs[?OutputKey=='StartupBucketName'].OutputValue" --output text)
DISTRIBUTION_ID=$(aws cloudformation describe-stacks --stack-name medimind-entry --region us-east-1 --query "Stacks[0].Outputs[?OutputKey=='DistributionId'].OutputValue" --output text)
aws s3 sync infrastructure/on-demand/site/ "s3://$STARTUP_BUCKET/" --delete --cache-control 'public,max-age=300'
aws cloudfront create-invalidation --distribution-id "$DISTRIBUTION_ID" --paths '/*'
```

### 5. Issue the EC2 origin certificate

Confirm `app.medimind-ai.online` resolves to the new Elastic IP while the old Nginx container is still serving the ACME webroot. The helper exists on this feature branch before the production checkout is changed, so run it from a temporary clone of this branch or execute the equivalent reviewed Certbot command through SSM. Do not deploy the new Nginx configuration until this command has succeeded:

```bash
TEMP_DIR=$(mktemp -d)
git clone --depth 1 --branch codex/on-demand-ec2-architecture \
  https://github.com/hammadAsher100/Medimind---Ai-powered-Healthcare-platform.git "$TEMP_DIR/source"
sudo CERTBOT_EMAIL=your-address@example.com \
  "$TEMP_DIR/source/medimind-ai/ops/prepare-on-demand-origin-certificate.sh"
```

The certificate remains at `/opt/medimind/certbot/conf/live/app.medimind-ai.online/` and is never committed.

### 6. Deploy application integration

Set these values in the persistent production `.env` (the state-aware workflow performs the same update after it is enabled):

```dotenv
ACTIVITY_TRACKING_ENABLED=True
ACTIVITY_TABLE_NAME=STACK_OUTPUT_TABLE_NAME
EC2_INSTANCE_ID=INSTANCE_ID
AWS_REGION=ap-south-1
ACTIVITY_TOUCH_INTERVAL_SECONDS=30
ACTIVITY_LEASE_SECONDS=900
CERTBOT_CERT_NAME=app.medimind-ai.online
```

Deploy the branch, verify `/readyz`, install the boot units, and perform a controlled EC2 stop/start test before DNS cutover.

### 7. Enable state-aware GitHub deployment

Add these GitHub production environment variables:

```text
ON_DEMAND_DEPLOYMENT_ENABLED=true
ACTIVITY_TABLE_NAME=<ActivityTableName stack output>
```

The regional stack attaches the required EC2/DynamoDB permissions to the existing GitHub OIDC role. Its existing S3 and SSM permissions remain unchanged.

### 8. Cut over the public entry domain

After the stopped-instance wake test succeeds, update the DNS stack so its existing root and `www` records change in place from the EC2 Elastic IP to CloudFront:

```bash
aws cloudformation deploy \
  --stack-name medimind-dns \
  --region ap-south-1 \
  --template-file infrastructure/on-demand/dns-template.yaml \
  --parameter-overrides \
    OriginElasticIp=ELASTIC_IP \
    EntryDistributionDomain=CLOUDFRONT_DOMAIN \
    CutoverEntryDns=true
```

Root and `www` will then point to CloudFront, while `app` continues to point directly to the Elastic IP.

Finally, update the regional stack with `AutomationEnabled=true` to activate the five-minute idle evaluation and periodic certificate-maintenance wake.

## Runtime behavior

- The startup browser calls `POST /control/wake` once and polls `GET /control/status` every 5–12 seconds.
- EC2 is started only in the `stopped` state. A wake arriving while `stopping` remains recorded and is reconciled when the next status poll sees `stopped`.
- A redirect is returned only after `https://app.medimind-ai.online/readyz` validates Django, PostgreSQL, FastAPI, all four tabular models, and the pneumonia CNN.
- Authenticated Django activity updates the table. Mutating Django and FastAPI requests create 15-minute TTL leases.
- The idle Lambda runs every five minutes and uses a conditional shutdown lock. Active leases, recent activity, minimum runtime, and deployments all prevent shutdown.
- A normal `StopInstances` request triggers systemd shutdown. `medimind-compose.service` gives Compose up to five minutes to stop containers cleanly.
- A maintenance wake every 20 days lets the persistent systemd certificate timer renew the Let's Encrypt origin certificate even during long periods without visitors.

## Troubleshooting

- **Startup page reports an error:** inspect `/aws/lambda/medimind-wake-status` in CloudWatch Logs.
- **EC2 runs but never becomes ready:** inspect `systemctl status medimind-compose`, `docker compose ps`, and `docker compose logs` through SSM.
- **FastAPI readiness fails:** verify `/opt/medimind/models/{diabetes,heart,kidney,stroke}` and the pneumonia HDF5 model.
- **Application certificate error:** check the `app.medimind-ai.online` certificate and `medimind-certbot-renew.timer`.
- **Instance does not stop:** inspect the DynamoDB `STATE` item and non-expired `LEASE#` items; deployments and active requests intentionally block shutdown.
- **Deployment cannot start EC2:** confirm the `MediMindStateAwareDeployment` inline policy is attached to the GitHub OIDC role.

## Data safety

The shutdown path never terminates EC2 and never invokes `docker compose down -v`. EBS, PostgreSQL, report media, Qdrant, Grafana, MLflow artifacts, certificates, `.env`, and model files remain attached and billable while EC2 is stopped.
