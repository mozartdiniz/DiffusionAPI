#!/bin/bash

# Generate self-signed SSL certificate for containerized environments
echo "Generating self-signed SSL certificate for container..."

# Generate private key
openssl genrsa -out key.pem 2048

# Create a config file for the certificate (localhost only)
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

# Generate certificate with Subject Alternative Names
openssl req -new -x509 -key key.pem -out cert.pem -days 365 -config openssl.conf

# Clean up config file
rm openssl.conf

echo "SSL certificates generated successfully!"
echo "key.pem - Private key"
echo "cert.pem - Certificate (localhost only - suitable for containers)"
echo ""
echo "You can now start the server with HTTPS using: ./init.sh"
echo "Note: For containers, access via localhost:7866 or 127.0.0.1:7866"
echo "For external access, you may need to configure your container networking." 