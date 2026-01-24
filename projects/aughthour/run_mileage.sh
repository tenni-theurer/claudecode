#!/bin/bash
# Run the mileage tracker script

# Set up environment for cron
export HOME="/Users/tenni"
export PATH="/Library/Frameworks/Python.framework/Versions/3.12/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

cd "$(dirname "$0")"

# Load environment variables from .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 mileage_tracker.py "$@"
