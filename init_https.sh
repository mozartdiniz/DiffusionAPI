#!/bin/bash

echo "DiffusionAPI Server Startup"
echo "=========================="
echo "1. Start with HTTP (default)"
echo "2. Start with HTTPS (self-signed certificate)"
echo "3. Start with HTTPS (custom certificates)"
echo ""
read -p "Choose an option (1-3): " choice

case $choice in
    1)
        echo "Starting server with HTTP..."
        uvicorn diffusionapi.main:app --host 0.0.0.0 --port 7866
        ;;
    2)
        echo "Starting server with HTTPS (self-signed certificate)..."
        # Check if certificates exist, generate if not
        if [ ! -f "key.pem" ] || [ ! -f "cert.pem" ]; then
            echo "Self-signed certificates not found. Generating..."
            ./generate_ssl_certs.sh
        fi
        uvicorn diffusionapi.main:app --host 0.0.0.0 --port 7866 --ssl-keyfile=./key.pem --ssl-certfile=./cert.pem
        ;;
    3)
        echo "Starting server with HTTPS (custom certificates)..."
        read -p "Enter path to private key file: " keyfile
        read -p "Enter path to certificate file: " certfile
        uvicorn diffusionapi.main:app --host 0.0.0.0 --port 7866 --ssl-keyfile="$keyfile" --ssl-certfile="$certfile"
        ;;
    *)
        echo "Invalid option. Starting with HTTP..."
        uvicorn diffusionapi.main:app --host 0.0.0.0 --port 7866
        ;;
esac 