"""
Network configuration for PDF to Excel Extractor
Run this script to start the server in network mode
"""

import streamlit as st
import socket
import subprocess
import sys

def get_network_ip():
    """Get the local network IP address"""
    try:
        # Create a socket to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"

def start_network_server():
    """Start Streamlit server in network mode"""
    local_ip = get_network_ip()
    port = 8505  # Use a different port to avoid conflicts
    
    print(f"🌐 Starting PDF to Excel Extractor in network mode...")
    print(f"📍 Server will be accessible at: http://{local_ip}:{port}")
    print(f"🔗 Local access: http://localhost:{port}")
    print("=" * 60)
    
    # Start Streamlit with network configuration
    cmd = [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", str(port),
        "--server.address", "0.0.0.0",  # Allow network access
        "--server.headless", "true",   # Run in headless mode
        "--server.enableCORS", "false", # Disable CORS for local network
        "--server.enableXsrfProtection", "false" # Disable XSRF for simplicity
    ]
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    start_network_server()