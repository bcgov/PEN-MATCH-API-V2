import os
from azure.keyvault.secrets import SecretClient
from azure.identity import ManagedIdentityCredential

class Settings:
    def __init__(self, key_vault_url=None):
        self.key_vault_url = key_vault_url or "https://pen-match-api-v2.vault.azure.net"
        
        if self.key_vault_url:
            credential = ManagedIdentityCredential()
            self.secret_client = SecretClient(vault_url=self.key_vault_url, credential=credential)
            
            # Load from Azure Key Vault
            self.tenant_url = self.get_secret("TENANT-URL")
            self.client_id = self.get_secret("CLIENT-ID")
            self.client_secret = self.get_secret("CLIENT-SECRET")
            self.api_base_url = self.get_secret("API-BASE-URL")
            self.openai_api_key = self.get_secret("OPENAI-API-KEY")
            self.openai_api_base = self.get_secret("OPENAI-API-BASE", required=False)
            self.cosmos_endpoint = self.get_secret("COSMOS-ENDPOINT")
            self.cosmos_key = self.get_secret("COSMOS-KEY")
            
        else:
            # Fallback to environment variables
            from dotenv import load_dotenv
            load_dotenv()
            self.tenant_url = os.getenv("TENANT_URL")
            self.client_id = os.getenv("CLIENT_ID")
            self.client_secret = os.getenv("CLIENT_SECRET")
            self.api_base_url = os.getenv("API_BASE_URL")
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            self.openai_api_base = os.getenv("OPENAI_API_BASE")
            self.cosmos_endpoint = os.getenv("COSMOS_ENDPOINT")
            self.cosmos_key = os.getenv("COSMOS_KEY")
            
    def get_secret(self, secret_name, required=True):
        try:
            retrieved_secret = self.secret_client.get_secret(secret_name)
            return retrieved_secret.value
        except Exception as e:
            if required:
                raise ValueError(f"Failed to retrieve secret '{secret_name}': {str(e)}")
            return None

settings = Settings()