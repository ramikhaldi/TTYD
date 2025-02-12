sudo systemctl stop ttyd
sudo systemctl disable ttyd
sudo rm /etc/systemd/system/ttyd.service
sudo systemctl daemon-reload
