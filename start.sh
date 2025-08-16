#!/bin/bash

# Railway startup script for Streamlit
export PORT=${PORT:-8080}
echo "Starting Streamlit on port $PORT"

# Start Streamlit with proper configuration
streamlit run main.py \
  --server.port $PORT \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false
