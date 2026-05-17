#!/usr/bin/env bash
# One-command environment setup for the PCM Thermal Forecasting project.
#
# Creates a local virtual environment under ./.venv and installs the pinned
# dependencies. Run from the project root:
#
#     bash setup.sh
#
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d .venv ]; then
    echo ">> Creating virtual environment at .venv ..."
    python3 -m venv .venv
fi

echo ">> Upgrading pip ..."
./.venv/bin/pip install --quiet --upgrade pip

echo ">> Installing dependencies from requirements.txt ..."
./.venv/bin/pip install --quiet -r requirements.txt

echo ""
echo "Environment ready. Activate it with:"
echo "    source .venv/bin/activate"
echo ""
echo "Or run scripts directly:"
echo "    .venv/bin/python shared/prepare_data.py"
echo "    .venv/bin/python experiments/01_ridge/tune.py"
echo "    .venv/bin/python experiments/02_random_forest/tune.py"
echo "    .venv/bin/python experiments/03_gradient_boosting/tune.py"
