#!/bin/bash

SERVICE_NAME="ttyd"
SERVICE_FILE="/etc/systemd/system/$SERVICE_NAME.service"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"  # Get the directory of the installer script
SCRIPT_PATH="$SCRIPT_DIR/start.sh"  # Set the full path to start.sh
USER=$(whoami)

echo "Installing $SERVICE_NAME service..."
echo "Script Path: $SCRIPT_PATH"

# Ensure start.sh exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: start.sh not found in $SCRIPT_DIR!"
    exit 1
fi

# Create the systemd service file
sudo bash -c "cat > $SERVICE_FILE" <<EOL
[Unit]
Description=TTYD Startup Script
After=network.target

[Service]
Type=simple
User=$USER
ExecStart=/bin/bash $SCRIPT_PATH
Restart=always
WorkingDirectory=$SCRIPT_DIR

[Install]
WantedBy=multi-user.target
EOL

# Reload systemd
echo "Reloading systemd..."
sudo systemctl daemon-reload

# Enable and start the service
echo "Enabling and starting the service..."
sudo systemctl enable $SERVICE_NAME
sudo systemctl start $SERVICE_NAME

echo "Installation complete!"
echo "Check status with: sudo systemctl status $SERVICE_NAME"
