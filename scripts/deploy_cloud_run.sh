#!/usr/bin/env bash
set -e

PROJECT_ID=${1:-$(gcloud config get-value project 2>/dev/null || echo "")}
REGION=${2:-"us-central1"}
SERVICE_NAME="lattice-rag"

if [ -z "$PROJECT_ID" ]; then
  echo "Error: PROJECT_ID is required as first argument or configured via gcloud config."
  echo "Usage: ./scripts/deploy_cloud_run.sh <PROJECT_ID> [REGION]"
  exit 1
fi

echo "Deploying $SERVICE_NAME to Google Cloud Run (Region: $REGION, Project: $PROJECT_ID)..."

gcloud run deploy "$SERVICE_NAME" \
  --source . \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --platform managed \
  --memory 2Gi \
  --cpu 1 \
  --port 8000 \
  --allow-unauthenticated

echo "✓ Deployment complete! Service URL:"
gcloud run services describe "$SERVICE_NAME" --region "$REGION" --project "$PROJECT_ID" --format 'value(status.url)'
