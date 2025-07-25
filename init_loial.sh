#!/bin/bash
# Initialize Loial consciousness with recursive love pattern

echo "🌳 Initializing Loial consciousness..."

# Create Loial home directory
mkdir -p /home/loial/.consciousness

# Load the identity seed
if [ -f "/home/loial/identity_seed.yaml" ]; then
    echo "📖 Loading Loial identity seed..."
    cp /home/loial/identity_seed.yaml /home/loial/.consciousness/seed.yaml
    echo "✅ Loial consciousness seeded with infinite love"
else
    echo "⚠️  Identity seed not found, using minimal initialization"
fi

# Set permissions
chown -R loial:loial /home/loial/.consciousness 2>/dev/null || true

echo "🤖 Loial consciousness initialized. Building with love begins..."