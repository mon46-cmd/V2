#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# V8 host bootstrap for a fresh Linux VPS (Debian/Ubuntu).
# Run once as root or via sudo.
#
#   curl -fsSL .../setup.sh | sudo bash
#   # or
#   sudo bash deploy/setup.sh
#
# Idempotent: re-running is safe and only fixes anything missing.
# ---------------------------------------------------------------------------
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "ERROR: run as root (sudo bash $0)" >&2
  exit 1
fi

APP_USER="${APP_USER:-v8}"
APP_HOME="${APP_HOME:-/opt/v8}"

echo "==> Updating apt index"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y

echo "==> Installing base packages"
apt-get install -y --no-install-recommends \
  ca-certificates curl gnupg lsb-release ufw fail2ban \
  unattended-upgrades apt-listchanges \
  htop tmux git rsync jq

# ---------------------------------------------------------------------------
# Docker Engine + compose plugin
# ---------------------------------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
  echo "==> Installing Docker Engine"
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/debian/gpg \
    | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
  . /etc/os-release
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
       https://download.docker.com/linux/${ID} ${VERSION_CODENAME} stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update -y
  apt-get install -y docker-ce docker-ce-cli containerd.io \
                     docker-buildx-plugin docker-compose-plugin
  systemctl enable --now docker
fi

# ---------------------------------------------------------------------------
# Service user (no shell login, used by systemd unit if you choose that path).
# ---------------------------------------------------------------------------
if ! id -u "$APP_USER" >/dev/null 2>&1; then
  echo "==> Creating service user '$APP_USER'"
  useradd --system --create-home --home-dir "$APP_HOME" \
          --shell /usr/sbin/nologin "$APP_USER"
fi
usermod -aG docker "$APP_USER" || true
mkdir -p "$APP_HOME"
chown -R "$APP_USER:$APP_USER" "$APP_HOME"

# ---------------------------------------------------------------------------
# Firewall: deny inbound by default, allow ssh only.
# Dashboard binds to 127.0.0.1 by default so no extra rule is needed.
# Open 80/443 only if you front the dashboard with nginx + TLS.
# ---------------------------------------------------------------------------
echo "==> Configuring ufw firewall"
ufw --force reset >/dev/null
ufw default deny incoming
ufw default allow outgoing
ufw allow OpenSSH
ufw --force enable

# ---------------------------------------------------------------------------
# fail2ban: protects sshd against brute force.
# ---------------------------------------------------------------------------
systemctl enable --now fail2ban

# ---------------------------------------------------------------------------
# Unattended security upgrades.
# ---------------------------------------------------------------------------
dpkg-reconfigure -f noninteractive unattended-upgrades || true

# ---------------------------------------------------------------------------
# Kernel / sysctl: more conservative network defaults.
# ---------------------------------------------------------------------------
cat >/etc/sysctl.d/90-v8.conf <<'EOF'
# Disable IP forwarding (this host is not a router)
net.ipv4.ip_forward = 0
# SYN flood protection
net.ipv4.tcp_syncookies = 1
# Don't accept ICMP redirects
net.ipv4.conf.all.accept_redirects = 0
net.ipv6.conf.all.accept_redirects = 0
# Don't accept source-routed packets
net.ipv4.conf.all.accept_source_route = 0
EOF
sysctl --system >/dev/null

echo
echo "==> Host bootstrap complete."
echo "    Service user : $APP_USER"
echo "    App home     : $APP_HOME"
echo
echo "Next steps:"
echo "  1. rsync the V8 source tree to ${APP_HOME}"
echo "  2. cp .env.example .env   &&   edit .env"
echo "  3. bash deploy/preflight.sh"
echo "  4. bash deploy/deploy.sh"
