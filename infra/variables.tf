# -------------
# Common Variables for Azure Infrastructure
# -------------

variable "vnet_name" {
  description = "Name of the existing virtual network"
  type        = string
}

variable "vnet_resource_group_name" {
  description = "Resource group where the virtual network exists"
  type        = string
}

variable "azure_openai_api_key" {
  description = "Azure OpenAI API key"
  type        = string
  sensitive   = true
}



variable "resource_group_name" {
  description = "Name of the existing resource group"
  type        = string
}

variable "api_image" {
  description = "The image for the API container"
  type        = string
}

variable "app_env" {
  description = "Application environment (dev, test, prod)"
  type        = string
}

variable "app_name" {
  description = "Name of the application"
  type        = string
}

variable "app_service_sku_name_backend" {
  description = "SKU name for the backend App Service Plan"
  type        = string
  default     = "B2" # Basic tier 
}

# Azure OpenAI Configuration
variable "azure_openai_endpoint" {
  description = "Azure OpenAI service endpoint URL"
  type        = string
  sensitive   = true
}

variable "azure_openai_deployment_name" {
  description = "Azure OpenAI model deployment name"
  type        = string
  default     = "gpt-4o"
}

variable "azure_openai_embedding_deployment" {
  description = "Azure OpenAI embedding model deployment name"
  type        = string
  default     = "text-embedding-3-large"
}

variable "common_tags" {
  description = "Common tags to apply to all resources"
  type        = map(string)
}

variable "image_tag" {
  description = "Tag for the container images"
  type        = string
  default     = "latest"
}

variable "location" {
  description = "Azure region for resources"
  type        = string
  default     = "Canada Central"
}



variable "repo_name" {
  description = "Name of the repository, used for resource naming"
  type        = string
  nullable    = false
}

variable "use_oidc" {
  description = "Use OIDC for authentication"
  type        = bool
  default     = true
}

variable "vnet_address_space" {
  type        = string
  description = "Address space for the virtual network, it is created by platform team"
}

variable "azure_openai_llm_endpoint" {
  description = "The endpoint for the Azure OpenAI LLM service."
  type        = string
  nullable    = false
}
variable "azure_openai_embedding_endpoint" {
  description = "The endpoint for the Azure OpenAI embedding service."
  type        = string
  nullable    = false
}



variable "web_subnet_cidr" {
  type = string
  description = "CIDR for the web subnet"
}

variable "app_service_subnet_cidr" {
  type = string
  description = "CIDR for the app service subnet"
}

variable "container_instance_subnet_cidr" {
  type = string
  description = "CIDR for the container instance subnet"
}


