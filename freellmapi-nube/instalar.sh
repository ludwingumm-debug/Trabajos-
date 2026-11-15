#!/usr/bin/env bash
# Instala FreeLLMAPI en un servidor Linux (Ubuntu/Debian) y lo publica SOLO
# dentro de tu red privada de Tailscale, con HTTPS.
#
# Uso (en el servidor):
#   bash instalar.sh
#
# Se puede volver a ejecutar sin peligro: conserva tu .env (y tu clave de
# cifrado) y actualiza el contenedor a la última versión.
set -euo pipefail

DIR="${FREELLMAPI_DIR:-$HOME/freellmapi}"
PORT="${PORT:-3001}"

paso() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

if [ "$(id -u)" -eq 0 ]; then SUDO=""; else SUDO="sudo"; fi

paso "1/5 Docker"
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | $SUDO sh
fi
if [ -n "$SUDO" ]; then $SUDO usermod -aG docker "$USER" || true; fi
command -v openssl >/dev/null 2>&1 || { $SUDO apt-get update -y && $SUDO apt-get install -y openssl; }

paso "2/5 Tailscale"
if ! command -v tailscale >/dev/null 2>&1; then
  curl -fsSL https://tailscale.com/install.sh | $SUDO sh
fi
if ! $SUDO tailscale status >/dev/null 2>&1; then
  echo "Abre el enlace que aparece abajo e inicia sesión con TU cuenta de Tailscale"
  echo "(la misma que usarás en tu ordenador y tu celular)."
  $SUDO tailscale up
fi

paso "3/5 Archivos de FreeLLMAPI en $DIR"
mkdir -p "$DIR"
cd "$DIR"

if [ ! -f .env ]; then
  umask 077
  printf 'ENCRYPTION_KEY=%s\nPORT=%s\nMCP_INFERENCE_DEFAULT_MODEL=auto\n' \
    "$(openssl rand -hex 32)" "$PORT" > .env
  echo "Creado .env con una clave de cifrado nueva."
  echo "IMPORTANTE: guarda una copia de $DIR/.env en un lugar seguro (gestor de contraseñas)."
else
  echo ".env ya existe: se conserva tal cual."
fi

cat > docker-compose.yml <<YAML
services:
  freellmapi:
    image: ghcr.io/tashfeenahmed/freellmapi:latest
    env_file: [.env]
    environment:
      NODE_ENV: production
      PORT: 3001
    # Solo en localhost: a internet NO se expone. Tailscale Serve lo publica
    # únicamente dentro de tu red privada.
    ports:
      - "127.0.0.1:${PORT}:3001"
    volumes:
      - freellmapi-data:/app/server/data
    extra_hosts:
      - "host.docker.internal:host-gateway"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "node", "-e", "fetch('http://127.0.0.1:3001/api/ping').then(r=>{if(!r.ok)process.exit(1)}).catch(()=>process.exit(1))"]
      interval: 30s
      timeout: 5s
      start_period: 15s
      retries: 3

volumes:
  freellmapi-data:
YAML

paso "4/5 Arrancando el contenedor"
$SUDO docker compose pull
$SUDO docker compose up -d

echo -n "Esperando a que responda"
for _ in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:${PORT}/api/ping" >/dev/null 2>&1; then echo " ¡listo!"; break; fi
  echo -n "."; sleep 2
done

paso "5/5 Publicando en tu red privada (Tailscale Serve, HTTPS)"
if ! $SUDO tailscale serve --bg "http://127.0.0.1:${PORT}"; then
  echo
  echo "Tailscale necesita que actives HTTPS en tu red una sola vez."
  echo "Abre el enlace que aparece arriba (o https://login.tailscale.com/admin/dns),"
  echo "activa 'MagicDNS' y 'HTTPS Certificates', y vuelve a ejecutar este script."
  exit 1
fi

NOMBRE="$($SUDO tailscale status --json | awk -F'"' '/"Self"/{s=1} s && /"DNSName"/{sub(/\.$/,"",$4); print $4; exit}')"
echo
echo "======================================================================"
echo " FreeLLMAPI está funcionando."
echo
echo " Panel (desde cualquier dispositivo con Tailscale encendido):"
echo "   https://${NOMBRE}"
echo
echo " URL para apps compatibles con OpenAI:   https://${NOMBRE}/v1"
echo " URL MCP (Claude Code):                   https://${NOMBRE}/mcp"
echo "======================================================================"
