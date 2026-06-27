#!/bin/bash

# ==========================================
# BACKGROUND SERVICES START (Docker & Desktop)
# ==========================================

# 1. Docker Daemon im Hintergrund starten (braucht als einziges sudo/root)
sudo dind dockerd > /tmp/dockerd.log 2>&1 &

# 2. VNC Server OHNE sudo starten (nutzt die Passwortdatei von user_lerobot)
vncserver :5 -geometry 1280x720 -depth 24 -localhost no

# 3. noVNC über Websockify starten (Der offizielle Ubuntu-Weg)
# Dies nimmt das VNC-Signal (5905) und wandelt es in eine Webseite auf Port 6080 um
websockify --web=/usr/share/novnc/ 6080 localhost:5905 > /tmp/novnc.log 2>&1 &

echo "=== Docker, VNC und noVNC (Port 6080) wurden im Hintergrund gestartet ==="

set -e
SSH_DIR="${HOME}/.ssh"

if [ -n "${SSH_PRIVATE_KEY}" ]; then
    mkdir -p "${SSH_DIR}"
    echo "${SSH_PRIVATE_KEY}" | base64 -d > "${SSH_DIR}/id_ed25519"
    chmod 600 "${SSH_DIR}/id_ed25519"
    unset SSH_PRIVATE_KEY
fi

mkdir -p "${SSH_DIR}"

if [ ! -f "${SSH_DIR}/known_hosts" ] || ! grep -q "github.com" "${SSH_DIR}/known_hosts" 2>/dev/null; then
    ssh-keyscan github.com >> "${SSH_DIR}/known_hosts" 2>/dev/null
fi

if [ ! -f "${SSH_DIR}/config" ]; then
    cat > "${SSH_DIR}/config" <<'EOF'
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
EOF
    chmod 600 "${SSH_DIR}/config"
fi

if [ ! -d /lerobot/.git ]; then
    if [ -z "${GIT_REPO_URL}" ]; then
        echo "Warning: GIT_REPO_URL is not set and /lerobot is not a git repo."
        echo "         Git operations will not be available."
    else
        cd /lerobot
        git init
        git remote add origin "${GIT_REPO_URL}"

        if [ -n "${GIT_TAG}" ]; then
            echo "Fetching repository: ${GIT_REPO_URL} (tag: ${GIT_TAG})"
            git fetch origin "refs/tags/${GIT_TAG}:refs/tags/${GIT_TAG}"
            git checkout -f "${GIT_TAG}"
        else
            BRANCH="${GIT_BRANCH:-main}"
            echo "Fetching repository: ${GIT_REPO_URL} (branch: ${BRANCH})"
            git fetch origin "${BRANCH}"
            git checkout -f -B "${BRANCH}" "origin/${BRANCH}"
        fi
    fi
else
    echo "Repository already present in /lerobot, skipping clone."
fi

cd /lerobot
if [ -f pyproject.toml ]; then
    echo "Installing project into virtual environment..."
    uv sync --locked --extra all
fi

exec "$@"