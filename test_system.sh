#!/bin/bash

cd /Users/rohanvinaik/REV

# Run the quick test to identify issues
python3 run_tests.py

# Run specific test files if the basic tests pass
if [ $? -eq 0 ]; then
    echo ""
    echo "Running unit tests..."
    python3 -m pytest tests/test_unified_system_simple.py -v -x
fi
