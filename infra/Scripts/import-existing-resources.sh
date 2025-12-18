#!/bin/bash

set -e

echo "=== Azure Resource Import Script ==="

# Function to safely import a resource
import_resource() {
    local terraform_address="$1"
    local azure_resource_id="$2"
    local resource_name="$3"
    
    echo "----------------------------------------"
    echo "Importing: $resource_name"
    echo "Terraform Address: $terraform_address"
    
    # Check if resource is already in state
    if terraform state show "$terraform_address" >/dev/null 2>&1; then
        echo "✓ Resource already in Terraform state: $resource_name"
        return 0
    fi
    
    # Check if Azure resource exists
    if ! az resource show --ids "$azure_resource_id" >/dev/null 2>&1; then
        echo "⚠ Azure resource does not exist: $resource_name"
        return 0
    fi
    
    # Attempt import with variable file
    echo "Importing resource with variables..."
    if terraform import -input=false -var-file="main.tfvars.json" "$terraform_address" "$azure_resource_id"; then
        echo "✓ Successfully imported: $resource_name"
    else
        echo "✗ Failed to import: $resource_name"
        return 1
    fi
}

# Import resources one by one
echo "Starting resource import process..."

import_resource \
    "module.backend.azurerm_service_plan.backend" \
    "/subscriptions/$ARM_SUBSCRIPTION_ID/resourceGroups/pen-match-api-v2/providers/Microsoft.Web/serverFarms/pen-match-api-v2-app-backend-asp" \
    "App Service Plan"

import_resource \
    "module.network.azurerm_network_security_group.privateendpoints" \
    "/subscriptions/$ARM_SUBSCRIPTION_ID/resourceGroups/$TF_VAR_vnet_resource_group_name/providers/Microsoft.Network/networkSecurityGroups/pen-match-api-v2-pe-nsg" \
    "Private Endpoints NSG"

import_resource \
    "module.network.azurerm_network_security_group.app_service" \
    "/subscriptions/$ARM_SUBSCRIPTION_ID/resourceGroups/$TF_VAR_vnet_resource_group_name/providers/Microsoft.Network/networkSecurityGroups/pen-match-api-v2-as-nsg" \
    "App Service NSG"

import_resource \
    "module.network.azurerm_network_security_group.web" \
    "/subscriptions/$ARM_SUBSCRIPTION_ID/resourceGroups/$TF_VAR_vnet_resource_group_name/providers/Microsoft.Network/networkSecurityGroups/pen-match-api-v2-web-nsg" \
    "Web NSG"

import_resource \
    "module.network.azurerm_network_security_group.container_instance" \
    "/subscriptions/$ARM_SUBSCRIPTION_ID/resourceGroups/$TF_VAR_vnet_resource_group_name/providers/Microsoft.Network/networkSecurityGroups/pen-match-api-v2-ci-nsg" \
    "Container Instance NSG"

echo "========================================="
echo "Import process completed!"
echo "Current Terraform state:"
terraform state list
echo "========================================= "