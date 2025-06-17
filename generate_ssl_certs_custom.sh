#!/bin/bash

# Generate self-signed SSL certificate for development with custom IP
echo "Generating self-signed SSL certificate..."

# Allow user to specify IP address
if [ -z "$1" ]; then
    echo "Usage: $0 <IP_ADDRESS>"
    echo "Example: $0 192.168.0.122"
    exit 1
fi

CUSTOM_IP=$1
echo "Using IP address: $CUSTOM_IP"

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
IP.2 = $CUSTOM_IP
EOF

# Generate certificate with Subject Alternative Names
openssl req -new -x509 -key key.pem -out cert.pem -days 365 -config openssl.conf

# Clean up config file
rm openssl.conf

echo "SSL certificates generated successfully!"
echo "key.pem - Private key"
echo "cert.pem - Certificate (includes localhost and $CUSTOM_IP)"
echo ""
echo "You can now start the server with HTTPS using: ./init.sh"
echo "Test with: curl --cacert cert.pem https://$CUSTOM_IP:7866/hello" 