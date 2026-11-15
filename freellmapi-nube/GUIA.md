# FreeLLMAPI en la nube — guía paso a paso

Objetivo: tener **FreeLLMAPI** funcionando 24/7 en un servidor en la nube y usarlo
desde tu **ordenador** y tu **celular**, de forma privada (solo tus dispositivos
pueden entrar).

```
 Celular ─┐                         ┌───────────── Servidor en la nube ─────────────┐
          ├── Tailscale (privado) ──┤ Tailscale Serve (HTTPS) → FreeLLMAPI :3001    │
 PC ──────┘                         │                     ↓                         │
                                    │        Groq, Gemini, OpenRouter, ...          │
                                    └───────────────────────────────────────────────┘
```

FreeLLMAPI **nunca** queda abierto a internet: solo escucha en `localhost` del
servidor, y Tailscale lo publica únicamente dentro de tu red privada.

---

## Paso 1 — Crear el servidor

Cualquier servidor Linux (Ubuntu 22.04/24.04) con **1 GB de RAM o más** sirve.

| Opción | Precio | Notas |
|---|---|---|
| **Oracle Cloud "Always Free"** | Gratis | VM Ampere (ARM) de hasta 4 CPU / 24 GB. El registro pide tarjeta para verificar. A veces no hay capacidad en tu región; reintenta. |
| **Hetzner** (CX22 / CAX11) | ~4 € / mes | Muy fiable y sencillo. |
| **DigitalOcean / Vultr** | ~5–6 USD / mes | Muy fáciles de usar. |

Al crearlo:
1. Elige **Ubuntu 24.04**.
2. Añade tu clave SSH (o anota la contraseña que te den).
3. **No** hace falta abrir ningún puerto aparte del 22 (SSH).

## Paso 2 — Crear tu cuenta de Tailscale e instalarlo en tus dispositivos

1. Crea una cuenta gratis en <https://tailscale.com> (puedes entrar con Google, Microsoft o GitHub).
2. Instala la app y **inicia sesión con esa misma cuenta** en:
   - tu **ordenador** (Windows / Mac / Linux): <https://tailscale.com/download>
   - tu **celular**: "Tailscale" en Google Play o App Store.
3. En el panel de Tailscale → **DNS** (<https://login.tailscale.com/admin/dns>):
   activa **MagicDNS** y **HTTPS Certificates**.

## Paso 3 — Instalar FreeLLMAPI en el servidor

Conéctate al servidor por SSH desde tu ordenador:

```bash
ssh ubuntu@IP_DE_TU_SERVIDOR      # en Hetzner/DigitalOcean suele ser root@...
```

Copia el script `instalar.sh` de esta carpeta al servidor y ejecútalo. La forma
más fácil es pegarlo:

```bash
nano instalar.sh      # pega el contenido, guarda con Ctrl+O, Enter, sal con Ctrl+X
bash instalar.sh
```

El script:
1. instala Docker y Tailscale,
2. te muestra un enlace para conectar el servidor a tu cuenta de Tailscale (ábrelo e inicia sesión),
3. crea `~/freellmapi/.env` con una clave de cifrado nueva,
4. descarga y arranca FreeLLMAPI,
5. lo publica con HTTPS dentro de tu red privada.

Al final verás algo como:

```
 Panel:                       https://mi-servidor.tail1234.ts.net
 URL para apps OpenAI:        https://mi-servidor.tail1234.ts.net/v1
 URL MCP (Claude Code):       https://mi-servidor.tail1234.ts.net/mcp
```

> 🔐 **Guarda una copia de `~/freellmapi/.env`** (por ejemplo en tu gestor de
> contraseñas). Ahí está la `ENCRYPTION_KEY` con la que se cifran tus claves de
> proveedores; si la pierdes, tendrás que volver a introducirlas todas.
>
> ```bash
> cat ~/freellmapi/.env
> ```

## Paso 4 — Primera configuración (desde el navegador)

Con Tailscale encendido en tu ordenador, abre la URL del panel.

1. **Crea tu cuenta** (email + contraseña). Es solo local, en tu servidor.
2. Página **Keys** → añade las claves de los proveedores gratuitos que quieras.
   Buenos para empezar (todos tienen plan gratuito):
   - Google AI Studio (Gemini) — <https://aistudio.google.com/apikey>
   - Groq — <https://console.groq.com/keys>
   - OpenRouter — <https://openrouter.ai/keys>
   - Cerebras, Mistral, GitHub Models, NVIDIA NIM… (el panel lista los 87)
3. **Fallback Chain** → ordena los modelos a tu gusto (o deja el orden automático).
4. Arriba en la página **Keys** está tu **clave unificada** (`freellmapi-...`).
   Esa es la única clave que usarás en tus apps.
5. (Opcional, para Claude Code) En **Keys → Agent compatibility**, activa el **MCP**.

## Paso 5 — Usarlo en tus dispositivos

> En todos los casos, **Tailscale tiene que estar encendido** en ese dispositivo.

### 📱 Celular
- **App oficial FreeLLMAPI** (Google Play / App Store): conéctala a la URL de tu panel.
- O cualquier app de chat que admita "proveedor compatible con OpenAI"
  (por ejemplo ChatBox, Pal Chat, etc.):
  - **Base URL:** `https://mi-servidor.tail1234.ts.net/v1`
  - **API key:** tu clave unificada `freellmapi-...`
  - **Modelo:** `auto`

### 💻 Ordenador

**Cualquier programa compatible con OpenAI** — misma Base URL y clave.

**Python:**
```python
from openai import OpenAI

client = OpenAI(
    base_url="https://mi-servidor.tail1234.ts.net/v1",
    api_key="freellmapi-TU-CLAVE",
)
r = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Hola, ¿qué modelo eres?"}],
)
print(r.choices[0].message.content)
```

**curl (prueba rápida):**
```bash
curl https://mi-servidor.tail1234.ts.net/v1/chat/completions \
  -H "Authorization: Bearer freellmapi-TU-CLAVE" \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"Hola"}]}'
```

**Claude Code** (como servidor MCP):
```bash
claude mcp add --transport http freellmapi https://mi-servidor.tail1234.ts.net/mcp \
  --header "Authorization: Bearer freellmapi-TU-CLAVE"
```

> ℹ️ claude.ai (web / app de Claude) no puede conectarse a este servidor como
> conector, porque es privado y usa una clave en cabecera. Desde Claude, úsalo a
> través de **Claude Code**.

---

## Mantenimiento

Todo se hace por SSH en el servidor:

| Qué | Comando |
|---|---|
| Actualizar a la última versión | `bash instalar.sh` (vuelve a ejecutarlo; conserva tus datos) |
| Ver si está funcionando | `cd ~/freellmapi && sudo docker compose ps` |
| Ver los logs | `cd ~/freellmapi && sudo docker compose logs -f --tail 100` |
| Reiniciar | `cd ~/freellmapi && sudo docker compose restart` |
| Ver la URL privada | `sudo tailscale serve status` |
| Copia de seguridad de la base de datos | `cd ~/freellmapi && sudo docker compose cp freellmapi:/app/server/data ./backup-$(date +%F)` |

**¿Olvidaste la contraseña del panel?** En la pantalla de login pulsa
*Forgot password? → Send reset code*, y busca el código en los logs:

```bash
cd ~/freellmapi && sudo docker compose logs --tail 50 | grep -i "reset code"
```

## Problemas frecuentes

- **La página no carga en el celular/PC** → comprueba que Tailscale está
  encendido en ese dispositivo y con la misma cuenta.
- **El script falla en el paso 5** → activa *MagicDNS* y *HTTPS Certificates* en
  <https://login.tailscale.com/admin/dns> y vuelve a ejecutar `bash instalar.sh`.
- **Un proveedor da error** → revisa la clave en el panel (página Keys); el
  router pasa automáticamente al siguiente proveedor de la cadena.

## Si más adelante lo quieres en tu ordenador

Solo hay que instalar la app de escritorio (Releases de
<https://github.com/tashfeenahmed/freellmapi>) y volver a introducir las claves,
o mover la base de datos del servidor. Avísame y lo preparamos.
