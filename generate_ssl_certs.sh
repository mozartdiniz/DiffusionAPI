#!/bin/bash

# Generate self-signed SSL certificate for development
echo "Generating self-signed SSL certificate..."

# Generate private key
openssl genrsa -out key.pem 2048

# Generate certificate
openssl req -new -x509 -key key.pem -out cert.pem -days 365 -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"

echo "SSL certificates generated successfully!"
echo "key.pem - Private key"
echo "cert.pem - Certificate"
echo ""
echo "You can now start the server with HTTPS using: ./init.sh"
echo "Note: Your browser will show a security warning for self-signed certificates."
echo "This is normal for development. Click 'Advanced' and 'Proceed to localhost' to continue." 