#!/bin/sh
set -eu
if [ "${DEMO_RESET_DATABASE_ON_START:-false}" = "true" ]; then
  python -m app.reset_demo_database
fi
exec "$@"
