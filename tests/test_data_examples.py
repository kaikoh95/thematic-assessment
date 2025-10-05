"""
Test Data Examples

Tests the Lambda handler against all example files in the data/ directory.
This ensures our implementation works with real-world data samples.

NOTE: The provided example data files contain duplicate IDs which fail validation.
These tests verify that validation correctly rejects invalid data.
"""

import pytest
import json
import os
from pathlib import Path
from src.lambda_function import handler


class MockContext:
    """Mock Lambda context for testing"""
    aws_request_id = "test-data-example"
    function_name = "text-analysis-test"
    memory_limit_in_mb = 3008


# Get path to data directory
DATA_DIR = Path(__file__).parent.parent / "data"


class TestDataExamples:
    """Test all example data files"""

    @pytest.mark.parametrize("example_file", [
        "input_example.json",
        "input_example_2.json",
        "input_comparison_example.json"
    ])
    def test_example_file_validation(self, example_file):
        """
        Test Lambda handler with example data files.

        NOTE: The provided example files contain duplicate IDs,
        so we expect 400 validation errors.
        """
        file_path = DATA_DIR / example_file

        # Skip if file doesn't exist
        if not file_path.exists():
            pytest.skip(f"Example file not found: {example_file}")

        # Load example data
        with open(file_path, 'r') as f:
            example_data = json.load(f)

        # Create Lambda event
        event = {
            "body": json.dumps(example_data)
        }

        # Invoke handler
        response = handler(event, MockContext())

        # The provided data files have duplicate IDs, so they should fail validation
        assert response['statusCode'] == 400, \
            f"{example_file} should fail validation due to duplicate IDs"

        body = json.loads(response['body'])
        assert 'error' in body
        assert 'validation' in body['error'].lower(), \
            f"Error should mention validation failure, got: {body['error']}"

        print(f"✓ {example_file}: Correctly rejected with validation error")

    def test_input_example_has_duplicates(self):
        """Verify input_example.json has duplicate IDs (as expected)"""
        file_path = DATA_DIR / "input_example.json"

        if not file_path.exists():
            pytest.skip("input_example.json not found")

        with open(file_path, 'r') as f:
            example_data = json.load(f)

        # Check for duplicates
        all_ids = [s['id'] for s in example_data.get('baseline', [])]
        unique_ids = set(all_ids)

        assert len(all_ids) > len(unique_ids), \
            "Expected input_example.json to have duplicate IDs"

        print(f"✓ Found {len(all_ids) - len(unique_ids)} duplicate IDs in baseline")

    def test_comparison_example_has_duplicates(self):
        """Verify input_comparison_example.json has duplicate IDs (as expected)"""
        file_path = DATA_DIR / "input_comparison_example.json"

        if not file_path.exists():
            pytest.skip("input_comparison_example.json not found")

        with open(file_path, 'r') as f:
            example_data = json.load(f)

        # Check baseline for duplicates
        baseline_ids = [s['id'] for s in example_data.get('baseline', [])]
        baseline_unique = set(baseline_ids)

        # Check comparison for duplicates
        comparison_ids = [s['id'] for s in example_data.get('comparison', [])]
        comparison_unique = set(comparison_ids)

        assert len(baseline_ids) > len(baseline_unique), \
            "Expected baseline to have duplicate IDs"

        assert len(comparison_ids) > len(comparison_unique), \
            "Expected comparison to have duplicate IDs"

        print(f"✓ Found {len(baseline_ids) - len(baseline_unique)} duplicate IDs in baseline")
        print(f"✓ Found {len(comparison_ids) - len(comparison_unique)} duplicate IDs in comparison")




class TestDataValidation:
    """Validate example data files meet expected format"""

    @pytest.mark.parametrize("example_file", [
        "input_example.json",
        "input_example_2.json",
        "input_comparison_example.json"
    ])
    def test_example_file_structure(self, example_file):
        """Validate example file has correct structure"""
        file_path = DATA_DIR / example_file

        if not file_path.exists():
            pytest.skip(f"Example file not found: {example_file}")

        with open(file_path, 'r') as f:
            data = json.load(f)

        # Must have baseline
        assert 'baseline' in data, f"Missing 'baseline' in {example_file}"
        assert isinstance(data['baseline'], list)
        assert len(data['baseline']) > 0, f"Empty baseline in {example_file}"

        # Each baseline sentence must have required fields
        for sentence in data['baseline']:
            assert 'sentence' in sentence
            assert 'id' in sentence
            assert isinstance(sentence['sentence'], str)
            assert isinstance(sentence['id'], str)
            assert len(sentence['sentence']) > 0
            assert len(sentence['id']) > 0

        # If comparison exists, validate it
        if 'comparison' in data and data['comparison'] is not None:
            assert isinstance(data['comparison'], list)

            # Only validate content if comparison is not empty
            if len(data['comparison']) > 0:
                for sentence in data['comparison']:
                    assert 'sentence' in sentence
                    assert 'id' in sentence
                    assert isinstance(sentence['sentence'], str)
                    assert isinstance(sentence['id'], str)
                    assert len(sentence['sentence']) > 0
                    assert len(sentence['id']) > 0


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
