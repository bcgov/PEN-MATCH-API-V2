"""
Simplified test for basic Azure OpenAI functionality with private endpoint support.
"""
import os
import asyncio
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

def test_basic_openai_connection():
    """Test basic connection to Azure OpenAI with enhanced error handling."""
    try:
        # Initialize Azure OpenAI LLM with additional configuration
        llm = AzureChatOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            openai_api_version="2024-12-01-preview",
            timeout=30,  # Increase timeout for private endpoints
            max_retries=3,  # Add retry logic
        )
        
        # Simple test message
        messages = [HumanMessage(content="Hello, respond with 'Connection successful' if you can see this.")]
        
        # Get response
        response = llm.invoke(messages)
        
        print("✅ Azure OpenAI Connection Test Passed")
        print(f"Response: {response.content}")
        return True
        
    except Exception as e:
        print(f"❌ Azure OpenAI Connection Test Failed: {str(e)}")
        
        # Enhanced error diagnostics
        if "403" in str(e):
            print("🔒 Network Access Issue Detected:")
            print("   - Check if public access is enabled in Azure Portal")
            print("   - Verify private endpoint configuration if using private networks")
            print("   - Ensure your IP is whitelisted if using selected networks")
        
        return False

def check_network_connectivity():
    """Check network connectivity to Azure OpenAI endpoint."""
    import urllib.request
    import urllib.error
    
    try:
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            print("❌ AZURE_OPENAI_ENDPOINT not configured")
            return False
            
        # Test basic connectivity
        req = urllib.request.Request(endpoint)
        response = urllib.request.urlopen(req, timeout=10)
        print(f"✅ Network connectivity to {endpoint} - Status: {response.status}")
        return True
        
    except urllib.error.HTTPError as e:
        if e.code == 403:
            print(f"🔒 Access denied (403) to {endpoint}")
            print("   This confirms the endpoint is reachable but access is restricted")
            return "restricted"
        else:
            print(f"❌ HTTP Error {e.code}: {e.reason}")
            return False
            
    except Exception as e:
        print(f"❌ Network connectivity test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Azure OpenAI diagnostic tests...\n")
    
    # Check environment variables
    required_vars = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT_NAME"]
    env_ok = all(os.environ.get(var) for var in required_vars)
    
    if not env_ok:
        print("❌ Missing environment variables")
        exit(1)
    
    print("✅ Environment variables configured\n")
    
    # Test network connectivity first
    connectivity = check_network_connectivity()
    print()
    
    if connectivity == "restricted":
        print("🔧 SOLUTION REQUIRED:")
        print("1. Enable public access in Azure Portal: OpenAI Resource → Networking → All Networks")
        print("2. OR configure private endpoint access from your current network")
        print("3. OR whitelist your current IP address in Azure OpenAI networking settings")
    elif connectivity:
        # Only proceed with OpenAI tests if network is accessible
        test_basic_openai_connection()