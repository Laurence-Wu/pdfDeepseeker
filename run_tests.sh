#!/bin/bash
##############################################################################
# PDF Translation Pipeline - VLA Integration Test Runner
# Run all integration tests for Instructions 00-07
##############################################################################

set -e  # Exit on error

echo "======================================================================"
echo "PDF TRANSLATION PIPELINE - VLA INTEGRATION TESTS"
echo "======================================================================"
echo ""
echo "Testing implementation of Instructions 00-07:"
echo "  - 03: GeminiClient with OpenRouter Integration"
echo "  - 04: PromptEngine with Advanced Prompts"
echo "  - 05: VLATrigger - Complexity Detection"
echo "  - 06: VLAProcessor - Model Integration"
echo "  - 07: VLAProcessingPipeline - Complete Pipeline"
echo ""
echo "======================================================================"
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Track results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to run a test
run_test() {
    local test_name="$1"
    local test_file="$2"

    echo "----------------------------------------------------------------------"
    echo "Running: $test_name"
    echo "----------------------------------------------------------------------"

    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    if python3 "$test_file" > /tmp/test_output.txt 2>&1; then
        echo -e "${GREEN}✅ PASSED${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))

        # Show summary
        grep -E "✅|passed|Results:" /tmp/test_output.txt | tail -3
    else
        echo -e "${RED}❌ FAILED${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))

        # Show error
        echo "Error output:"
        tail -20 /tmp/test_output.txt
    fi

    echo ""
}

# Run all tests
echo ""
run_test "1. GeminiClient Tests" "tests/integration/test_gemini_client.py"
run_test "2. PromptEngine Tests" "tests/integration/test_gemini_client_part2.py"
run_test "3. VLATrigger Tests" "tests/integration/test_vla_trigger.py"
run_test "4. VLAProcessor Tests" "tests/integration/test_vla_processor.py"
run_test "5. VLAProcessingPipeline Tests" "tests/integration/test_vla_pipeline.py"

# Summary
echo "======================================================================"
echo "TEST SUMMARY"
echo "======================================================================"
echo ""
echo "Total Test Suites: $TOTAL_TESTS"
echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}"
echo -e "Failed: ${RED}$FAILED_TESTS${NC}"
echo ""

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}✅ ALL TESTS PASSED!${NC}"
    echo ""
    echo "The VLA integration is working correctly."
    echo "See TEST_RESULTS.txt for detailed results."
    exit 0
else
    echo -e "${RED}❌ SOME TESTS FAILED${NC}"
    echo ""
    echo "Please check the error messages above."
    echo "See TEST_RESULTS.txt for troubleshooting guidance."
    exit 1
fi
