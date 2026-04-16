#!/bin/bash

# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python dependencies using uv
uv sync

# Create uploads folder if it doesn't exist
mkdir -p instance
mkdir -p uploads
