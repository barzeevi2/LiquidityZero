#!/bin/bash
# Simple script to run ingestor for 24 hours

cd "$(dirname "$0")/.."

# Activate virtual environment
source venv/bin/activate

# Start Docker services
docker-compose -f docker/docker-compose.yml up -d

# Wait a moment for services to start
sleep 3

# Run ingestor
python -m app.data.ingestor

