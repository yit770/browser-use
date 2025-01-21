"""
Simple test script for LMStudio API
"""

import requests
import json

def test_lmstudio():
    base_url = "http://192.168.100.164:1234/v1"
    
    # Test models endpoint
    print("\nTesting models endpoint...")
    try:
        response = requests.get(f"{base_url}/models")
        print(f"Status code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Test chat completion
    print("\nTesting chat completion...")
    try:
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello!"}
            ],
            "temperature": 0.7,
            "max_tokens": 512,
            "stream": False
        }
        
        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=data
        )
        
        print(f"Status code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_lmstudio()
