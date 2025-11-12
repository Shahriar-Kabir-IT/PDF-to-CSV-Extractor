#!/bin/bash

echo "========================================"
echo "PDF Table Extractor - Server Startup"
echo "========================================"
echo ""

# Get the local IP address (works on Linux/Mac)
LOCAL_IP=$(hostname -I | awk '{print $1}')

# If hostname -I doesn't work, try alternative methods
if [ -z "$LOCAL_IP" ]; then
    LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr eth0 2>/dev/null || echo "localhost")
fi

echo "Your local IP address: $LOCAL_IP"
echo ""
echo "Starting Streamlit server..."
echo ""
echo "Users on your network can access the app at:"
echo "http://$LOCAL_IP:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo "========================================"
echo ""

python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501

