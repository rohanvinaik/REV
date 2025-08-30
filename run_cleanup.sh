#!/bin/bash

cd /Users/rohanvinaik/REV

echo "Running REV project cleanup..."
python3 cleanup_project.py

echo ""
echo "Testing basic functionality..."
python3 run_tests.py

echo ""
echo "Cleanup and basic tests complete!"
