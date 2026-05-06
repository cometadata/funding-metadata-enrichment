#!/usr/bin/env python3
"""
Quick test script to verify the API is working correctly.

This script:
1. Checks if the API is running
2. Lists available sample documents
3. Processes one document and shows results
"""

import requests
from pathlib import Path
import sys
import json

API_BASE_URL = "http://localhost:8000"


def check_api_health():
    """Check if the API is running and healthy."""
    try:
        response = requests.get(f"{API_BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            print("✅ API is running")
            print(f"   - Status: {data['status']}")
            print(f"   - Model loaded: {data['model_loaded']}")
            print(f"   - Queries loaded: {data['queries_loaded']}")
            return True
        else:
            print("❌ API returned non-200 status")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API at", API_BASE_URL)
        print("   Please start the API with: uvicorn funding_api:app --reload")
        return False
    except Exception as e:
        print(f"❌ Error checking API: {e}")
        return False


def list_sample_documents():
    """List available sample documents."""
    sample_dir = Path("sample_documents")
    if not sample_dir.exists():
        print("❌ No sample_documents directory found")
        return []
    
    md_files = list(sample_dir.glob("*.md"))
    if md_files:
        print(f"\n📁 Found {len(md_files)} sample documents:")
        for f in md_files:
            print(f"   - {f.name}")
    else:
        print("❌ No markdown files found in sample_documents/")
    
    return md_files


def test_extraction(file_path: Path):
    """Test extraction on a single file."""
    print(f"\n🔍 Testing extraction on: {file_path.name}")
    print("-" * 50)
    
    with open(file_path, 'rb') as f:
        files = {'file': (file_path.name, f, 'text/markdown')}
        data = {'normalize': 'true'}
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/extract-funding",
                files=files,
                data=data
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Show summary
                summary = result['summary']
                print(f"✅ Extraction successful!")
                print(f"   - Statements found: {summary['total_statements']}")
                print(f"   - Unique statements: {summary['unique_statements']}")
                print(f"   - Processing time: {result['metadata']['processing_time']:.2f}s")
                
                # Show first statement if any
                if result['funding_statements']:
                    print(f"\n💰 First funding statement:")
                    stmt = result['funding_statements'][0]
                    print(f"   \"{stmt['statement'][:100]}...\"")
                    print(f"   Score: {stmt['score']:.2f}")
                
                return True
            else:
                print(f"❌ Extraction failed: HTTP {response.status_code}")
                print(f"   {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error during extraction: {e}")
            return False


def main():
    """Run quick test."""
    print("Funding Statement Extraction API - Quick Test")
    print("=" * 60)
    
    # Step 1: Check API health
    print("\n1. Checking API health...")
    if not check_api_health():
        sys.exit(1)
    
    # Step 2: List sample documents
    print("\n2. Checking sample documents...")
    sample_files = list_sample_documents()
    
    if not sample_files:
        print("\n⚠️  No sample documents found to test")
        sys.exit(1)
    
    # Step 3: Test extraction on first file
    print("\n3. Testing extraction...")
    test_file = sample_files[0]
    success = test_extraction(test_file)
    
    # Summary
    print("\n" + "=" * 60)
    if success:
        print("✅ All tests passed! The API is working correctly.")
        print("\nNext steps:")
        print("- Try: python basic_api_example.py")
        print("- Try: python websocket_example.py sample_documents/" + test_file.name)
        print("- Try: python batch_processing_example.py sample_documents/")
    else:
        print("❌ Tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()