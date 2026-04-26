# V8 — Linux VPS Deployment Runbook

This document is the single source of truth for running V8 on a Linux VPS
(Debian 12 / Ubuntu 22.04+). It covers first-time bootstrap, day-2 ops,
upgrades, monitoring and recovery.

> The trading engine is **paper-trading by default**. Going live requires
> explicit code changes in `src/portfolio/paper_broker.py` and is out of
> scope for this guide.

---

## 1. Architecture on the VPS

```
                       +--------------------------------+
                       |  VPS (single host, Docker)     |
                       |                                |
                       |  +--------+    +-----------+   |
                       |  | engine |--->| app_data  |<--|--+
                       |  +--------+    | (volume)  |   |  |
                       |                +-----------+   |  |
                       |  +-----------+      ^          |  |
   localhost:8765 <----+--| dashboard |------+          |  |
                       |  +-----------+                 |  |
                       +--------------------------------+  |
                                                            |
                              backup.sh ----- ./backups/ ---+
```

* **engine** — `python -m loops.runner`, no inbound network, writes to
  the shared `app_data` volume.
* **dashboard** — `python -m web_api.app`, FastAPI on port 8765 bound to
  `127.0.0.1` by default. Read-only view of state written by the engine.
* Both run from the **same image** (built once) and the **same `.env`**.

---

## 2. First-time host bootstrap

```bash
# As root on a fresh Debian/Ubuntu VPS:
git clone <your-private-repo> /opt/v8
cd /opt/v8
sudo bash deploy/setup.sh
```

`setup.sh` is idempotent and:

* installs Docker Engine + the compose plugin
* creates a non-login service user `v8` (uid auto-assigned)
* enables `ufw` (deny-incoming, ssh-only)
* enables `fail2ban`
* enables unattended security upgrades
* hardens basic sysctl

### SSH hardening (do this manually before exposing the host)

```bash
# /etc/ssh/sshd_config
PasswordAuthentication no
PermitRootLogin no
```

Then `sudo systemctl reload ssh`.

---

## 3. Configure the application

```bash
cd /opt/v8
cp .env.example .env
chmod 600 .env
$EDITOR .env
```

Required keys (preflight will warn otherwise):

| Key                  | Purpose                                |
|----------------------|----------------------------------------|
| `OPENROUTER_API_KEY` | All AI calls                           |
| `DAILY_BUDGET_USD`   | Hard cap on daily AI spend             |
| `MAX_DRAWDOWN_PCT`   | Portfolio-level halt (e.g. `0.05`)     |

Recommended defaults for production:

```dotenv
LOG_LEVEL=INFO
API_BIND=127.0.0.1        # never expose 0.0.0.0 without a TLS reverse proxy
DAILY_BUDGET_USD=1.00
MAX_DRAWDOWN_PCT=0.05
BAR_MAX_AGE_SEC=1200
```

---

## 4. Deploy

```bash
cd /opt/v8
bash deploy/preflight.sh   # verifies env, docker, ports, disk, clock
bash deploy/deploy.sh      # build + up + tail logs for 30s
```

Verify:

```bash
docker compose ps
docker compose logs -f engine
curl -s http://127.0.0.1:8765/health   # -> {"status":"ok"}
```

### Auto-start at boot (optional)

```bash
sudo cp deploy/systemd/v8.service /etc/systemd/system/v8.service
sudo systemctl daemon-reload
sudo systemctl enable --now v8.service
```

---

## 5. Day-2 operations

| Task                | Command                                                   |
|---------------------|-----------------------------------------------------------|
| Stream logs         | `docker compose logs -f`                                  |
| Restart engine only | `docker compose restart engine`                           |
| Restart all         | `docker compose restart`                                  |
| Stop everything     | `docker compose down`                                     |
| Wipe data (!)       | `docker compose down -v`                                  |
| Update code         | `git pull && bash deploy/deploy.sh`                       |
| Backup data         | `bash deploy/backup.sh`                                   |
| Inspect container   | `docker compose exec dashboard sh`                        |
| Disk usage          | `docker system df`                                        |
| Prune images        | `docker image prune -f`                                   |

### Remote dashboard access

The dashboard is bound to `127.0.0.1`. For remote access, use an SSH tunnel:

```bash
# from your laptop:
ssh -N -L 8765:127.0.0.1:8765 v8-vps
# then open http://localhost:8765
```

For permanent public access, terminate TLS at nginx/Caddy and proxy to
`127.0.0.1:8765`. Do **not** set `API_BIND=0.0.0.0` without a reverse
proxy + TLS.

---

## 6. Monitoring

* **Container health**: `docker compose ps` shows `healthy/unhealthy`.
* **Logs**: JSON-rotated at 10MB × 5 files per container. View with
  `docker compose logs --tail=200 engine`.
* **Live trading state**: `curl /api/status` returns positions, equity,
  budget consumption, drawdown.
* **Audit trail**: `data/runs/audit.jsonl` is the append-only ledger of
  every AI call (cost, prompt hash, response).

### Suggested external monitor

A 1-minute cron on a separate host:

```cron
* * * * * curl -fsS https://your-host/health || curl -X POST https://hooks.slack.com/...
```

---

## 7. Backups

```bash
# Manual:
bash deploy/backup.sh

# Daily at 04:30 UTC:
crontab -e -u v8
30 4 * * * cd /opt/v8 && bash deploy/backup.sh >> /var/log/v8-backup.log 2>&1
```

`backup.sh` keeps the last 14 archives. Ship them off-host (rsync, S3,
restic) — a backup that lives on the VPS is not a backup.

---

## 8. Recovery

### Engine crashed and won't restart

```bash
docker compose logs --tail=200 engine        # find the traceback
docker compose down                          # clear bad state
docker compose up -d
```

If the container is healthy but positions look stale:

```bash
docker compose exec engine ls -la /app/data/runs
# verify positions.json / equity.json are recent
```

### Restore from backup

```bash
docker compose down                # stop both services
docker volume rm v8_app_data       # destroy the bad volume
docker volume create v8_app_data
docker run --rm \
  -v v8_app_data:/data \
  -v "$PWD/backups":/backup \
  alpine:3 \
  sh -c "cd /data && tar xzf /backup/v8-data-YYYYmmdd-HHMMSS.tgz"
docker compose up -d
```

### Full host loss

1. Provision a new VPS, run `setup.sh`.
2. `git clone` the repo into `/opt/v8`.
3. Copy `.env` from your secret store; copy backup tarball.
4. Restore as above, then `bash deploy/deploy.sh`.

---

## 9. Security checklist

- [ ] SSH key-only, no root login, no password auth
- [ ] `ufw` enabled, only port 22 open inbound
- [ ] `.env` is `chmod 600`, owned by deploy user
- [ ] `OPENROUTER_API_KEY`, `BYBIT_*` rotated quarterly
- [ ] Dashboard NOT exposed publicly without TLS reverse proxy
- [ ] `fail2ban` active (`systemctl status fail2ban`)
- [ ] `unattended-upgrades` active
- [ ] Backups tested by performing a full restore at least once
- [ ] `DAILY_BUDGET_USD` set to a sane cap
- [ ] `MAX_DRAWDOWN_PCT` set (engine halts new entries on breach)
- [ ] Containers run as non-root (`uid 10001`), read-only rootfs,
      `cap_drop: ALL`, `no-new-privileges`
- [ ] System clock is NTP-synchronized (`timedatectl`)

---

## 10. Upgrades

```bash
cd /opt/v8
git fetch && git log --oneline HEAD..origin/main   # review changes
git pull
bash deploy/deploy.sh                              # rebuild + restart
docker image prune -f                              # reclaim disk
```

Roll back:

```bash
git checkout <previous-sha>
bash deploy/deploy.sh
```

---

## 11. Troubleshooting matrix

| Symptom                                       | Likely cause                                 | Fix                                                      |
|-----------------------------------------------|----------------------------------------------|----------------------------------------------------------|
| `dashboard` is `unhealthy`                    | App didn't start                             | `docker compose logs dashboard`; check `.env`            |
| `engine` restarts in a loop                   | Bad config or upstream API down              | `docker compose logs engine`; check `OPENROUTER_API_KEY` |
| `429` from OpenRouter                         | Daily budget exhausted or rate limit         | Wait for UTC rollover or raise `DAILY_BUDGET_USD`        |
| Empty positions after restart                 | Data volume wiped (`down -v`)                | Restore from `backups/`                                  |
| Healthcheck fails locally but `/health` 200s  | DNS resolution inside container              | Check `wget --tries=1 --spider`; ensure ipv4 default     |
| Disk full                                     | Docker image/log accumulation                | `docker system prune -af && docker volume prune`         |

---

## 12. Going live (paper -> real)

This is intentionally NOT a single switch. At minimum:

1. Replace `PaperBroker` with a real Bybit broker implementation in
   `src/portfolio/`.
2. Add ed25519/HMAC signing using `BYBIT_API_KEY` / `BYBIT_API_SECRET`.
3. Cap initial position size at the smallest tradable lot.
4. Run for at least 30 days with `DAILY_BUDGET_USD` low and observe
   `MAX_DRAWDOWN_PCT` engagement.
5. Add Telegram/PagerDuty alerts (`src/notify/` is a stub).
6. Ensure backup job runs and is tested.

Until then, **keep paper trading**.
