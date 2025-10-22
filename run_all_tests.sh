#!/bin/bash
#
# Run All Integration Tests for pdfDeepseeker
# Executes all test suites in sequence and reports results
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "============================================================"
echo "PDF Deepseeker - Running All Integration Tests"
echo "============================================================"
echo ""

# Track results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to run a test suite
run_test() {
    local test_name=$1
    local test_file=$2

    echo -e "${BLUE}Running: ${test_name}${NC}"
    echo "------------------------------------------------------------"

    if .venv/bin/python "$test_file"; then
        echo -e "${GREEN}✅ ${test_name} PASSED${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${RED}❌ ${test_name} FAILED${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi

    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo ""
}

# Run configuration tests
if [ -d "tests/configuration" ]; then
    echo -e "${YELLOW}Configuration Tests${NC}"
    run_test "API Endpoints" "tests/configuration/test_api_endpoints.py"
    run_test "Config Files" "tests/configuration/test_config_files.py"
    run_test "Environment" "tests/configuration/test_env.py"
    run_test "File System" "tests/configuration/test_file_system.py"
    run_test "OpenRouter" "tests/configuration/test_openrouter.py"
    run_test "Database" "tests/configuration/test_database.py"
    run_test "Redis Connection" "tests/configuration/test_redis.py"
    echo ""
fi

# Run integration tests
echo -e "${YELLOW}VLA Integration Tests${NC}"
run_test "VLA Trigger (Part 1)" "tests/integration/test_vla_trigger.py"
run_test "VLA Processor (Part 2)" "tests/integration/test_vla_processor.py"
run_test "VLA Pipeline (Part 3)" "tests/integration/test_vla_pipeline.py"
echo ""

# Final summary
echo "============================================================"
echo "Test Summary"
echo "============================================================"
echo -e "Total Test Suites:  ${TOTAL_TESTS}"
echo -e "Passed:             ${GREEN}${PASSED_TESTS}${NC}"
echo -e "Failed:             ${RED}${FAILED_TESTS}${NC}"
echo "============================================================"

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}✅ ALL TESTS PASSED!${NC}"
    exit 0
else
    echo -e "${RED}❌ SOME TESTS FAILED${NC}"
    exit 1
fi
