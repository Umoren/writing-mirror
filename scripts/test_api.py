#!/usr/bin/env python3
"""
Test script for the Voice Writing Assistant API

This script tests the suggest API endpoint to ensure it's working correctly.
Run this after starting the FastAPI server with: uvicorn app.main:app --reload --port 8001

Filename: scripts/test_api.py
Purpose: Comprehensive testing of all API endpoints with real LLM integration
"""

import requests
import json
import time
from typing import Dict, Any
import sys
import os

# API Configuration
API_BASE_URL = "http://localhost:8001"
SUGGEST_ENDPOINT = f"{API_BASE_URL}/api/suggest"
HEALTH_ENDPOINT = f"{API_BASE_URL}/api/health"
API_HEALTH_ENDPOINT = f"{API_BASE_URL}/api/health"  # The detailed health check
STATUS_ENDPOINT = f"{API_BASE_URL}/api/status"


def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_test(test_name: str):
    """Print a formatted test header"""
    print(f"\n🔍 {test_name}")
    print("-" * 50)


def test_basic_connectivity():
    """Test basic API connectivity"""
    print_test("Testing Basic API Connectivity")
    
    try:
        response = requests.get(API_BASE_URL, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API is running: {data.get('app')} v{data.get('version')}")
            print(f"   Status: {data.get('status')}")
            return True
        else:
            print(f"❌ API returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Failed to connect to API: {str(e)}")
        print("   Make sure the server is running with: uvicorn app.main:app --reload --port 8001")
        return False


def test_health_endpoints():
    """Test all health check endpoints"""
    print_test("Testing Health Check Endpoints")
    
    # Test basic health endpoint
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Basic health check passed")
        else:
            print(f"❌ Basic health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Basic health check error: {str(e)}")
        return False
    
    # Test detailed API health endpoint
    try:
        response = requests.get(API_HEALTH_ENDPOINT, timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Detailed health check passed: {health_data.get('status')}")
            
            # Print service status details
            services = health_data.get('services', {})
            for service_name, service_info in services.items():
                if isinstance(service_info, dict):
                    status = service_info.get('status', 'unknown')
                    print(f"   {service_name}: {status}")
                    if 'model' in service_info:
                        print(f"      Model: {service_info['model']}")
                else:
                    print(f"   {service_name}: {service_info}")
            return True
        else:
            print(f"❌ Detailed health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Detailed health check error: {str(e)}")
        return False


def test_api_status():
    """Test the API status endpoint"""
    print_test("Testing API Status Endpoint")
    
    try:
        response = requests.get(STATUS_ENDPOINT, timeout=10)
        if response.status_code == 200:
            status_data = response.json()
            print(f"✅ API status check passed: {status_data.get('status')}")
            print(f"   Version: {status_data.get('version')}")
            
            services = status_data.get('services', {})
            for service, status in services.items():
                print(f"   {service}: {status}")
            return True
        else:
            print(f"❌ API status check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API status error: {str(e)}")
        return False


def test_suggest_endpoint(request_data: Dict[str, Any], test_name: str):
    """Test the suggest endpoint with given request data"""
    print_test(f"Testing Suggest Endpoint: {test_name}")
    
    print(f"Input text: '{request_data['text']}'")
    print(f"Context: {request_data.get('context', 'None')}")
    print(f"Task: {request_data.get('task', 'continue')}")
    print(f"Requesting {request_data.get('num_suggestions', 3)} suggestions...")
    
    try:
        start_time = time.time()
        response = requests.post(
            SUGGEST_ENDPOINT,
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=60  # LLM calls can take time
        )
        end_time = time.time()
        
        response_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Request successful (took {response_time:.0f}ms)")
            print(f"   Trace ID: {data.get('trace_id')}")
            
            # Print suggestions
            suggestions = data.get('suggestions', [])
            print(f"   Generated {len(suggestions)} suggestions:")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"      {i}. \"{suggestion['text']}\"")
                print(f"         Score: {suggestion.get('score', 0):.3f}")
                print(f"         Reasoning: {suggestion.get('reasoning', 'N/A')}")
            
            # Print source information
            sources = data.get('sources', [])
            if sources:
                print(f"   Retrieved from {len(sources)} source documents:")
                for source in sources[:3]:  # Show first 3 sources
                    print(f"      - {source.get('title', 'Unknown')} (similarity: {source.get('similarity', 0):.3f})")
            
            # Print performance stats
            stats = data.get('stats', {})
            print(f"   Performance breakdown:")
            print(f"      Total time: {stats.get('total_time_ms', 0)}ms")
            print(f"      Embedding: {stats.get('embedding_time_ms', 0)}ms")
            print(f"      Search: {stats.get('search_time_ms', 0)}ms")
            print(f"      Generation: {stats.get('generation_time_ms', 0)}ms")
            print(f"      Chunks retrieved: {stats.get('chunks_retrieved', 0)}")
            
            # Quality checks
            if len(suggestions) == request_data.get('num_suggestions', 3):
                print("   ✅ Correct number of suggestions returned")
            else:
                print(f"   ⚠️  Expected {request_data.get('num_suggestions', 3)} suggestions, got {len(suggestions)}")
            
            if all(len(s['text'].strip()) > 0 for s in suggestions):
                print("   ✅ All suggestions contain text")
            else:
                print("   ❌ Some suggestions are empty")
            
            return True
            
        else:
            print(f"❌ Request failed with status {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data}")
            except:
                print(f"   Raw response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Request error: {str(e)}")
        return False


def run_suggestion_tests():
    """Run a comprehensive set of suggestion tests"""
    print_test("Running Comprehensive Suggestion Tests")
    
    test_cases = [
        {
            "data": {
                "text": "I was thinking about machine learning and how it",
                "context": "Technical blog post about AI",
                "task": "continue",
                "num_suggestions": 3,
                "max_length": 100
            },
            "name": "Technical Content Continuation"
        },
        {
            "data": {
                "text": "The most important thing to remember when writing",
                "context": "Writing advice",
                "task": "complete",
                "num_suggestions": 2,
                "max_length": 80
            },
            "name": "Writing Advice Completion"
        },
        {
            "data": {
                "text": "Today was a productive day because",
                "context": "Personal journal entry",
                "task": "continue",
                "num_suggestions": 1,
                "max_length": 50
            },
            "name": "Personal Writing Continuation"
        },
        {
            "data": {
                "text": "Building software requires careful planning and attention to detail",
                "context": "Technical documentation",
                "task": "rephrase",
                "num_suggestions": 2,
                "max_length": 120
            },
            "name": "Technical Documentation Rephrase"
        },
        {
            "data": {
                "text": "When I started this project, I never imagined",
                "context": "Reflection on a completed project",
                "task": "continue",
                "num_suggestions": 2,
                "max_length": 90
            },
            "name": "Project Reflection Continuation"
        }
    ]
    
    successful_tests = 0
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}/{len(test_cases)} ---")
        if test_suggest_endpoint(test_case["data"], test_case["name"]):
            successful_tests += 1
        
        # Small delay between tests to be nice to the API
        if i < len(test_cases):
            time.sleep(2)
    
    return successful_tests, len(test_cases)


def main():
    """Run all API tests"""
    print_section("Voice Writing Assistant API Test Suite")
    print("Testing the suggest API endpoint with LLM integration")
    print(f"Target API: {API_BASE_URL}")
    
    # Track test results
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Basic connectivity
    total_tests += 1
    if test_basic_connectivity():
        tests_passed += 1
    else:
        print("\n❌ Basic connectivity failed. Cannot continue testing.")
        return False
    
    # Test 2: Health checks
    total_tests += 1
    if test_health_endpoints():
        tests_passed += 1
    
    # Test 3: API status
    total_tests += 1
    if test_api_status():
        tests_passed += 1
    
    # Test 4: Suggestion tests
    if tests_passed == total_tests:  # Only run if basic tests pass
        successful_suggestions, total_suggestions = run_suggestion_tests()
        tests_passed += successful_suggestions
        total_tests += total_suggestions
    else:
        print("\n⚠️  Skipping suggestion tests due to basic test failures")
    
    # Final summary
    print_section("Test Results Summary")
    print(f"Tests passed: {tests_passed}/{total_tests}")
    print(f"Success rate: {(tests_passed/total_tests)*100:.1f}%")
    
    if tests_passed == total_tests:
        print("\n🎉 All tests passed! The API is working correctly.")
        print("Your Voice Writing Assistant is ready to generate personalized suggestions!")
        return True
    else:
        print(f"\n⚠️  {total_tests - tests_passed} test(s) failed.")
        print("Check the error messages above for troubleshooting.")
        return False


if __name__ == "__main__":
    print("🚀 Starting Voice Writing Assistant API Tests...")
    success = main()
    
    if success:
        print("\n✅ Testing complete - API is fully functional!")
    else:
        print("\n❌ Testing complete - Some issues need to be resolved.")
    
    sys.exit(0 if success else 1)