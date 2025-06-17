#!/bin/bash

# Generate self-signed SSL certificate for development
echo "Generating self-signed SSL certificate..."

# Get the local IP address - try multiple methods for containerized environments
LOCAL_IP=""
if command -v ifconfig >/dev/null 2>&1; then
    LOCAL_IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | head -n 1)
elif command -v ip >/dev/null 2>&1; then
    LOCAL_IP=$(ip route get 1.1.1.1 | awk '{print $7}' | head -n 1)
elif command -v hostname >/dev/null 2>&1; then
    # Fallback to hostname -I if available
    LOCAL_IP=$(hostname -I | awk '{print $1}' 2>/dev/null)
fi

# If we still don't have an IP, use a default or ask user
if [ -z "$LOCAL_IP" ]; then
    echo "Could not automatically detect IP address."
    echo "Available options:"
    echo "1. Use localhost only (recommended for containers)"
    echo "2. Enter IP address manually"
    echo "3. Use 0.0.0.0 (bind to all interfaces)"
    read -p "Choose option (1-3): " choice
    
    case $choice in
        1)
            LOCAL_IP=""
            echo "Using localhost only"
            ;;
        2)
            read -p "Enter IP address: " LOCAL_IP
            ;;
        3)
            LOCAL_IP="0.0.0.0"
            ;;
        *)
            LOCAL_IP=""
            echo "Using localhost only"
            ;;
    esac
else
    echo "Detected local IP: $LOCAL_IP"
fi

# Generate private key
openssl genrsa -out key.pem 2048

# Create a config file for the certificate
cat > openssl.conf << EOF
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req
prompt = no

[req_distinguished_name]
C = US
ST = State
L = City
O = Organization
CN = localhost

[v3_req]
keyUsage = keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
DNS.2 = 127.0.0.1
IP.1 = 127.0.0.1
EOF

# Add the detected IP to the certificate if it exists and is valid
if [ -n "$LOCAL_IP" ] && [ "$LOCAL_IP" != "0.0.0.0" ]; then
    echo "IP.2 = $LOCAL_IP" >> openssl.conf
fi

# Generate certificate with Subject Alternative Names
openssl req -new -x509 -key key.pem -out cert.pem -days 365 -config openssl.conf

# Clean up config file
rm openssl.conf

echo "SSL certificates generated successfully!"
echo "key.pem - Private key"
if [ -n "$LOCAL_IP" ]; then
    echo "cert.pem - Certificate (includes localhost and $LOCAL_IP)"
else
    echo "cert.pem - Certificate (includes localhost only)"
fi
echo ""
echo "You can now start the server with HTTPS using: ./init.sh"
echo "Note: Your browser will show a security warning for self-signed certificates."
echo "This is normal for development. Click 'Advanced' and 'Proceed to localhost' to continue." 