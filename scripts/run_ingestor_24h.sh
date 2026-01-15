#!/bin/bash
# Simple script to run ingestor for 24 hours

cd "$(dirname "$0")/.."

# Activate virtual environment
source venv/bin/activate

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker daemon is not running."
    echo "Please start Docker Desktop and try again."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose > /dev/null 2>&1; then
    echo "Error: docker-compose is not installed or not in PATH."
    exit 1
fi

# Start Docker services
echo "Starting Docker services..."
if ! docker-compose -f docker/docker-compose.yml up -d; then
    echo "Error: Failed to start Docker services."
    exit 1
fi

# Wait a moment for services to start
echo "Waiting for services to initialize..."
sleep 3

# Run ingestor
python -m app.data.ingestor

