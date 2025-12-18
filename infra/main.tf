# -------------
# Root Level Terraform Configuration
# -------------
# Create the main resource group for all application resources

# resource "azurerm_resource_group" "main" {
#   name     = var.resource_group_name
#   location = var.location
#   tags     = var.common_tags
#   lifecycle {
#     ignore_changes = [
#       tags
#     ]
#   }
# }

# Data source to reference existing resource group if needed
data "azurerm_resource_group" "main" {
  name = var.resource_group_name
}

# -------------
# Modules based on Dependency
# -------------
module "network" {
  source = "./modules/network"

  common_tags              = var.common_tags
  resource_group_name      = data.azurerm_resource_group.main.name
  vnet_address_space       = var.vnet_address_space
  vnet_name                = var.vnet_name
  vnet_resource_group_name = var.vnet_resource_group_name

  depends_on = [data.azurerm_resource_group.main]
}



module "backend" {
  source = "./modules/backend"

  api_image                               = var.api_image
  app_env                                 = var.app_env
  app_name                                = var.app_name
  app_service_sku_name_backend            = var.app_service_sku_name_backend
  app_service_subnet_id                   = module.network.app_service_subnet_id
  azure_openai_endpoint                   = var.azure_openai_endpoint
  azure_openai_api_key                    = var.azure_openai_api_key
  azure_openai_deployment_name            = var.azure_openai_deployment_name
  azure_openai_embedding_deployment       = var.azure_openai_embedding_deployment
  backend_subnet_id                       = module.network.app_service_subnet_id
  common_tags                             = var.common_tags
  frontend_possible_outbound_ip_addresses = ""
  location                                = var.location
  private_endpoint_subnet_id              = module.network.private_endpoint_subnet_id
  repo_name                               = var.repo_name
  resource_group_name                     = data.azurerm_resource_group.main.name
  image_tag                               = var.image_tag
  azure_openai_embedding_endpoint         = var.azure_openai_embedding_endpoint
  azure_openai_llm_endpoint               = var.azure_openai_llm_endpoint

}


