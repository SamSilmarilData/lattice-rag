#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID=${1:-$(gcloud config get-value project 2>/dev/null || echo "")}
REGION=${2:-"us-central1"}
SERVICE_NAME="lattice-rag"

if [ -z "$PROJECT_ID" ]; then
  echo "Error: PROJECT_ID is required as first argument or configured via gcloud config."
  echo "Usage: ./scripts/deploy_cloud_run.sh <PROJECT_ID> [REGION]"
  exit 1
fi

# API keys are read from Secret Manager, never baked into the image or passed on the CLI.
# Create them once with: printf '%s' "$VALUE" | gcloud secrets create NAME --data-file=-
REQUIRED_SECRETS=(TYPESAFE_API_KEY GROQ_API_KEY GEMINI_API_KEY ADMIN_API_KEY)
for secret in "${REQUIRED_SECRETS[@]}"; do
  if ! gcloud secrets describe "$secret" --project "$PROJECT_ID" >/dev/null 2>&1; then
    echo "Error: Secret Manager secret '$secret' does not exist in project $PROJECT_ID."
    exit 1
  fi
done
SECRET_FLAGS=$(for s in "${REQUIRED_SECRETS[@]}"; do printf '%s=%s:latest,' "$s" "$s"; done | sed 's/,$//')

echo "Deploying $SERVICE_NAME to Google Cloud Run (Region: $REGION, Project: $PROJECT_ID)..."

# The LatticeDB file lives on the instance's ephemeral filesystem. Pin to a single instance so
# every request sees the same graph; documents ingested at runtime are lost on restart or
# redeploy (the image ships pre-seeded with the golden corpus).
# The service is publicly reachable; /api/v1/ingest, /eval, /benchmark, /graph and /cache require X-Admin-Key.
gcloud run deploy "$SERVICE_NAME" \
  --source . \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --platform managed \
  --memory 2Gi \
  --cpu 1 \
  --port 8000 \
  --min-instances 0 \
  --max-instances 1 \
  --set-secrets "$SECRET_FLAGS" \
  --allow-unauthenticated

echo "✓ Deployment complete! Service URL:"
gcloud run services describe "$SERVICE_NAME" --region "$REGION" --project "$PROJECT_ID" --format 'value(status.url)'
