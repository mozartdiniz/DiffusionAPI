#!/bin/bash

# Verify and fix SSL certificate for RunPod
echo "Verifying SSL certificate..."

# Check if certificate exists
if [ ! -f "cert.pem" ]; then
    echo "Certificate not found. Generating new one..."
    ./generate_ssl_certs.sh
    exit 0
fi

# Get the current IP
CURRENT_IP=$(ip route get 1.1.1.1 | awk '{print $7}' | head -n 1)
echo "Current IP: $CURRENT_IP"

# Check what's in the certificate
echo "Certificate details:"
openssl x509 -in cert.pem -text -noout | grep -A 10 "Subject Alternative Name"

# Regenerate certificate with the correct IP
echo ""
echo "Regenerating certificate with IP: $CURRENT_IP"

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
IP.2 = $CURRENT_IP
EOF

# Generate certificate with Subject Alternative Names
openssl req -new -x509 -key key.pem -out cert.pem -days 365 -config openssl.conf

# Clean up config file
rm openssl.conf

echo "Certificate regenerated successfully!"
echo "Testing certificate..."

# Test the certificate
if curl --cacert cert.pem https://$CURRENT_IP:7866/hello 2>/dev/null; then
    echo "✅ Certificate works correctly!"
else
    echo "❌ Certificate still has issues. Trying alternative approach..."
    
    # Alternative: Generate certificate with IP as CN
    openssl genrsa -out key.pem 2048
    openssl req -new -x509 -key key.pem -out cert.pem -days 365 -subj "/C=US/ST=State/L=City/O=Organization/CN=$CURRENT_IP"
    
    echo "Certificate regenerated with IP as CN. Testing..."
    if curl --cacert cert.pem https://$CURRENT_IP:7866/hello 2>/dev/null; then
        echo "✅ Certificate works with IP as CN!"
    else
        echo "❌ Still having issues. Consider using HTTP for development."
    fi
fi 