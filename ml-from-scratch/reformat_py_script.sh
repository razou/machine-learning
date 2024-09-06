#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <python_script.py>"
    exit 1
fi

# The target Python script
TARGET_SCRIPT=$1

# Function to prompt and execute a command
run_tool() {
    local description=$1
    local command=$2
    echo "$description"
    read -p "Hit 1 to proceed or 0 to skip: " choice

    if [ "$choice" -eq 1 ]; then
        echo "Running: $command"
        eval "$command"
    else
        echo "Skipping..."
    fi
}

# Define the commands with their descriptions, using the TARGET_SCRIPT variable
run_tool "1. Black, Purpose: Automatically formats Python code according to PEP 8 standards." "black $TARGET_SCRIPT"
run_tool "2. isort, Purpose: Sorts and organizes imports, automatically placing them in the correct order and removing unused imports." "isort $TARGET_SCRIPT"
run_tool "3. autoflake, Purpose: Removes unused imports and variables from your code." "autoflake --remove-all-unused-imports --remove-unused-variables --in-place $TARGET_SCRIPT"
run_tool "4. docformatter, Purpose: Reformats and adds consistent docstrings to your code." "docformatter -i $TARGET_SCRIPT"
run_tool "5. pydocstyle, Purpose: Checks that your Python docstrings comply with PEP 257." "pydocstyle $TARGET_SCRIPT"

echo "All selected operations are complete."
