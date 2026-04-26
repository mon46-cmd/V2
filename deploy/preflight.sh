#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Pre-deploy verification. Exits non-zero on any blocker.
#
# Run from the V8 project root:
#   bash deploy/preflight.sh
# ---------------------------------------------------------------------------
set -euo pipefail

ok()   { printf "  \e[32mOK\e[0m  %s\n" "$1"; }
warn() { printf "  \e[33mWARN\e[0m %s\n" "$1"; }
fail() { printf "  \e[31mFAIL\e[0m %s\n" "$1"; exit 1; }

echo "==> Preflight checks"

# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------
[[ -f .env ]]               || fail ".env not found (cp .env.example .env)"
[[ -f Dockerfile ]]         || fail "Dockerfile missing"
[[ -f docker-compose.yml ]] || fail "docker-compose.yml missing"
ok "core files present"

# ---------------------------------------------------------------------------
# Required env keys
# ---------------------------------------------------------------------------
set -a; . ./.env; set +a

required=(OPENROUTER_API_KEY DAILY_BUDGET_USD MAX_DRAWDOWN_PCT)
missing=0
for k in "${required[@]}"; do
  v="${!k:-}"
  if [[ -z "$v" ]]; then
    warn "$k is empty"
    missing=$((missing+1))
  fi
done
[[ $missing -eq 0 ]] || fail "$missing required env var(s) missing"
ok "required env keys set"

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
command -v docker >/dev/null 2>&1 || fail "docker not installed"
docker info >/dev/null 2>&1       || fail "docker daemon not reachable (try: sudo systemctl start docker)"
docker compose version >/dev/null 2>&1 || fail "'docker compose' plugin missing"
ok "docker engine + compose ready"

# ---------------------------------------------------------------------------
# Disk space (need at least 5GB on the volume hosting /var/lib/docker)
# ---------------------------------------------------------------------------
free_gb=$(df -BG --output=avail /var/lib/docker 2>/dev/null | tail -n1 | tr -dc '0-9' || echo 0)
if [[ -z "$free_gb" || "$free_gb" -lt 5 ]]; then
  warn "less than 5GB free on /var/lib/docker (have ${free_gb:-?}GB)"
else
  ok "${free_gb}GB free on /var/lib/docker"
fi

# ---------------------------------------------------------------------------
# Port availability
# ---------------------------------------------------------------------------
port="${API_PORT:-8765}"
bind="${API_BIND:-127.0.0.1}"
if ss -lnt "( sport = :$port )" 2>/dev/null | grep -q ":$port"; then
  warn "port $port already in use on $bind (compose will fail to start dashboard)"
else
  ok "port $port free on $bind"
fi

# ---------------------------------------------------------------------------
# Time sync (correct clock matters for exchange auth)
# ---------------------------------------------------------------------------
if command -v timedatectl >/dev/null 2>&1; then
  if timedatectl show -p NTPSynchronized --value | grep -q yes; then
    ok "system clock NTP-synchronized"
  else
    warn "system clock not NTP-synchronized (run: sudo timedatectl set-ntp true)"
  fi
fi

echo
echo "==> Preflight passed."
