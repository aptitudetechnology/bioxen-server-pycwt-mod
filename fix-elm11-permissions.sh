#!/usr/bin/env bash
# Fix permissions for ELM11 USB serial devices

echo "Fixing USB serial port permissions..."
echo ""

# Check which devices exist
if [ -e /dev/ttyUSB0 ]; then
    echo "Setting permissions on /dev/ttyUSB0..."
    sudo chmod 666 /dev/ttyUSB0
fi

if [ -e /dev/ttyUSB1 ]; then
    echo "Setting permissions on /dev/ttyUSB1..."
    sudo chmod 666 /dev/ttyUSB1
fi

# Add user to dialout group for permanent access (requires logout/login)
echo ""
echo "Adding user to 'dialout' group for permanent serial port access..."
sudo usermod -a -G dialout $USER

echo ""
echo "✓ Permissions fixed!"
echo ""
echo "Note: For permanent access, log out and log back in (to apply dialout group)"
echo ""
echo "Now run: python test-elm11.py"
