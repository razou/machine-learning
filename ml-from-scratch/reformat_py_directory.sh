#!/bin/bash

# Check if the directory parameter is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <directory_path>"
    exit 1
fi

# The target directory
TARGET_DIR=$1

# Function to prompt and execute a command on all .py files
run_tool() {
    local description=$1
    local command=$2
    echo "$description"
    read -p "Hit 1 to proceed or 0 to skip: " choice

    if [ "$choice" -eq 1 ]; then
        echo "Running: $command on all .py files in $TARGET_DIR"
        for TARGET_SCRIPT in "$TARGET_DIR"/*.py; do
            echo "Processing $TARGET_SCRIPT..."
            eval "$command $TARGET_SCRIPT"
        done
    else
        echo "Skipping..."
    fi
}

# Define the commands with their descriptions
run_tool "1. Black, Purpose: Automatically formats Python code according to PEP 8 standards." "black"
run_tool "2. isort, Purpose: Sorts and organizes imports, automatically placing them in the correct order and removing unused imports." "isort"
run_tool "3. autoflake, Purpose: Removes unused imports and variables from your code." "autoflake --remove-all-unused-imports --remove-unused-variables --in-place"
run_tool "4. docformatter, Purpose: Reformats and adds consistent docstrings to your code." "docformatter -i"
run_tool "5. pydocstyle, Purpose: Checks that your Python docstrings comply with PEP 257." "pydocstyle"

echo "All selected operations for all files are complete."
