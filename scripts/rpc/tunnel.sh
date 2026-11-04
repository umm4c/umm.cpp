#!/usr/bin/env bash
# Foreground SSH tunnel: use a dedicated terminal or your process supervisor.
set -euo pipefail
[[ $# == 1 ]] || { echo "Usage: $0 SSH_HOST_ALIAS" >&2; exit 2; }
: "${RPC_LOCAL_PORT:=50052}" "${RPC_REMOTE_PORT:=50052}"
[[ $RPC_LOCAL_PORT =~ ^[0-9]+$ && $RPC_REMOTE_PORT =~ ^[0-9]+$ ]] || exit 2
exec ssh -N -T -o BatchMode=yes -o ExitOnForwardFailure=yes \
    -o StrictHostKeyChecking=yes -o UpdateHostKeys=no -o ConnectTimeout=10 \
    -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    -L "127.0.0.1:$RPC_LOCAL_PORT:127.0.0.1:$RPC_REMOTE_PORT" -- "$1"
