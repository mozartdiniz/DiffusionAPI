#!/bin/bash

echo "Quick SSL Fix for RunPod"
echo "========================"
echo ""
echo "Current situation: SSL certificate mismatch with IP 172.19.0.2"
echo ""
echo "Available solutions:"
echo "1. Use HTTP instead of HTTPS (easiest)"
echo "2. Access via localhost instead of IP"
echo "3. Regenerate certificate with correct IP"
echo "4. Disable SSL verification in your client"
echo ""

read -p "Choose option (1-4): " choice

case $choice in
    1)
        echo "Starting server with HTTP..."
        echo "Stop the current server (Ctrl+C) and run:"
        echo "uvicorn diffusionapi.main:app --host 0.0.0.0 --port 7866"
        echo ""
        echo "Then access via: http://172.19.0.2:7866"
        ;;
    2)
        echo "Access via localhost instead of IP..."
        echo "Test with: curl --cacert cert.pem https://localhost:7866/hello"
        echo "Or: curl --cacert cert.pem https://127.0.0.1:7866/hello"
        ;;
    3)
        echo "Regenerating certificate with correct IP..."
        ./verify_and_fix_cert.sh
        ;;
    4)
        echo "To disable SSL verification in your client:"
        echo ""
        echo "For curl:"
        echo "curl -k https://172.19.0.2:7866/hello"
        echo ""
        echo "For Node.js/Fastify:"
        echo "const https = require('https');"
        echo "const httpsAgent = new https.Agent({ rejectUnauthorized: false });"
        echo "// Use with axios or fetch"
        ;;
    *)
        echo "Invalid option. Using HTTP..."
        echo "Stop the current server (Ctrl+C) and run:"
        echo "uvicorn diffusionapi.main:app --host 0.0.0.0 --port 7866"
        ;;
esac 