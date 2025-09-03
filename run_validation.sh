#!/bin/bash
# REV Comprehensive Validation Runner
# This script runs the complete validation suite locally

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PYTHON="${PYTHON:-python3}"
VALIDATION_DIR="experiments/validation_report"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_DIR="${VALIDATION_DIR}_${TIMESTAMP}"

# Functions
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

check_requirements() {
    print_header "Checking Requirements"
    
    # Check Python version
    if command -v $PYTHON &> /dev/null; then
        PY_VERSION=$($PYTHON --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
        print_success "Python $PY_VERSION found"
    else
        print_error "Python not found. Please install Python 3.9+"
        exit 1
    fi
    
    # Check if requirements are installed
    if $PYTHON -c "import numpy" 2>/dev/null; then
        print_success "NumPy installed"
    else
        print_warning "NumPy not found. Installing requirements..."
        $PYTHON -m pip install -r requirements.txt
    fi
    
    # Check for pandas, matplotlib, sklearn (needed for validation)
    MISSING_DEPS=""
    for dep in pandas matplotlib sklearn; do
        if ! $PYTHON -c "import $dep" 2>/dev/null; then
            MISSING_DEPS="$MISSING_DEPS $dep"
        fi
    done
    
    if [ -n "$MISSING_DEPS" ]; then
        print_warning "Installing validation dependencies:$MISSING_DEPS"
        $PYTHON -m pip install pandas matplotlib scikit-learn seaborn
    fi
    
    print_success "All requirements satisfied"
}

run_unit_tests() {
    print_header "Running Unit Tests"
    
    if [ -d "tests" ]; then
        if command -v pytest &> /dev/null; then
            pytest tests/ -v --tb=short --maxfail=5 \
                --junit-xml=${REPORT_DIR}/junit.xml \
                --cov=src --cov-report=xml:${REPORT_DIR}/coverage.xml \
                --cov-report=term || {
                print_warning "Some unit tests failed"
            }
        else
            print_warning "pytest not found. Skipping unit tests."
            print_warning "Install with: pip install pytest pytest-cov"
        fi
    else
        print_warning "No tests directory found"
    fi
}

run_validation_suite() {
    print_header "Running Comprehensive Validation Suite"
    
    # Create report directory
    mkdir -p $REPORT_DIR
    
    # Run comprehensive validation
    echo "Report will be saved to: $REPORT_DIR"
    
    # Override output directory in script
    export VALIDATION_OUTPUT_DIR=$REPORT_DIR
    
    $PYTHON experiments/comprehensive_validation.py || {
        print_error "Validation suite failed"
        exit 1
    }
    
    print_success "Validation suite completed"
}

check_documentation() {
    print_header "Checking Documentation"
    
    DOC_ISSUES=""
    
    # Check if documentation files exist
    for doc in docs/ARCHITECTURE.md docs/USER_GUIDE.md docs/API_REFERENCE.md; do
        if [ -f "$doc" ]; then
            print_success "$doc exists"
        else
            print_warning "$doc missing"
            DOC_ISSUES="$DOC_ISSUES $doc"
        fi
    done
    
    # Check example scripts
    EXAMPLE_COUNT=$(find examples -name "*.py" 2>/dev/null | wc -l)
    if [ "$EXAMPLE_COUNT" -gt 0 ]; then
        print_success "Found $EXAMPLE_COUNT example scripts"
        
        # Syntax check
        for script in examples/*.py; do
            if [ -f "$script" ]; then
                $PYTHON -m py_compile "$script" 2>/dev/null || {
                    print_warning "Syntax error in $script"
                    DOC_ISSUES="$DOC_ISSUES $script"
                }
            fi
        done
    else
        print_warning "No example scripts found"
    fi
    
    if [ -z "$DOC_ISSUES" ]; then
        print_success "Documentation check passed"
    else
        print_warning "Documentation issues found:$DOC_ISSUES"
    fi
}

run_security_checks() {
    print_header "Running Security Checks"
    
    # Check for bandit
    if command -v bandit &> /dev/null; then
        bandit -r src/ -f json -o ${REPORT_DIR}/security-bandit.json 2>/dev/null || {
            print_warning "Security issues detected by Bandit"
        }
        print_success "Bandit scan completed"
    else
        print_warning "Bandit not installed. Install with: pip install bandit"
    fi
    
    # Check for safety
    if command -v safety &> /dev/null; then
        safety check --json > ${REPORT_DIR}/security-safety.json 2>&1 || {
            print_warning "Vulnerable dependencies detected by Safety"
        }
        print_success "Safety check completed"
    else
        print_warning "Safety not installed. Install with: pip install safety"
    fi
}

generate_summary() {
    print_header "Generating Summary Report"
    
    SUMMARY_FILE="${REPORT_DIR}/summary.txt"
    
    cat > $SUMMARY_FILE << EOF
REV VALIDATION SUMMARY
======================
Date: $(date)
Report Directory: $REPORT_DIR

VALIDATION RESULTS
------------------
EOF
    
    # Check for validation results
    if [ -f "${REPORT_DIR}/executive_summary.txt" ]; then
        echo "" >> $SUMMARY_FILE
        cat "${REPORT_DIR}/executive_summary.txt" >> $SUMMARY_FILE
    fi
    
    # List all generated files
    echo -e "\nGENERATED FILES" >> $SUMMARY_FILE
    echo "---------------" >> $SUMMARY_FILE
    ls -la $REPORT_DIR >> $SUMMARY_FILE
    
    print_success "Summary report generated: $SUMMARY_FILE"
    
    # Display summary
    echo ""
    cat $SUMMARY_FILE
}

main() {
    print_header "REV Comprehensive Validation"
    echo "Starting validation at $(date)"
    echo "Report directory: $REPORT_DIR"
    
    # Run validation steps
    check_requirements
    check_documentation
    run_unit_tests
    run_security_checks
    run_validation_suite
    generate_summary
    
    print_header "Validation Complete"
    print_success "All validation reports saved to: $REPORT_DIR"
    
    # Open report if possible
    if [ -f "${REPORT_DIR}/report.html" ]; then
        if command -v open &> /dev/null; then
            print_success "Opening HTML report..."
            open "${REPORT_DIR}/report.html"
        elif command -v xdg-open &> /dev/null; then
            print_success "Opening HTML report..."
            xdg-open "${REPORT_DIR}/report.html"
        else
            print_success "View report at: ${REPORT_DIR}/report.html"
        fi
    fi
    
    # Check overall success
    if [ -f "${REPORT_DIR}/full_results.json" ]; then
        # Parse JSON to check success (simplified)
        if grep -q '"success": true' "${REPORT_DIR}/full_results.json"; then
            print_success "Validation PASSED"
            exit 0
        else
            print_warning "Validation completed with issues"
            exit 1
        fi
    fi
}

# Handle arguments
case "${1:-}" in
    --quick)
        print_warning "Quick mode: Skipping comprehensive tests"
        check_requirements
        check_documentation
        generate_summary
        ;;
    --help|-h)
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --quick    Run quick validation only"
        echo "  --help     Show this help message"
        echo ""
        echo "Environment Variables:"
        echo "  PYTHON     Python interpreter to use (default: python3)"
        echo ""
        exit 0
        ;;
    *)
        main
        ;;
esac