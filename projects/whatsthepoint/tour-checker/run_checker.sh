#!/bin/bash
# Run the tour checker script

# Set up environment for cron
export HOME="/Users/tenni"
export PATH="/Library/Frameworks/Python.framework/Versions/3.12/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
export PLAYWRIGHT_BROWSERS_PATH="$HOME/Library/Caches/ms-playwright"

cd "$(dirname "$0")"

# Load Gmail App Password from .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 check_tours.py "$@"
