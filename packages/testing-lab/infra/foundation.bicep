targetScope = 'resourceGroup'
param location string = 'westus2'
param prefix string = 'magnitude-lab'
param registryName string = 'magnitudelab5304'
param storageName string = 'magnitudelab5304c4b3'
var tags = { 'lab-component': 'coordinator' }

module network 'network.bicep' = {
  name: '${prefix}-network'
  params: { prefix: prefix, location: location, tags: tags }
}
resource registry 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: registryName
  location: location
  tags: tags
  sku: { name: 'Basic' }
  properties: { adminUserEnabled: false }
}
resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: '${prefix}-coordinator'
  location: location
  tags: tags
}
resource allocationRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(resourceGroup().id, identity.id, 'lab-allocation')
  properties: {
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'b24988ac-6180-42a0-ab88-20f7382dd24c')
  }
}
resource pullRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(registry.id, identity.id, 'lab-pull')
  scope: registry
  properties: {
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d')
  }
}
resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' existing = { name: storageName }
resource blobRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storage.id, identity.id, 'lab-blobs')
  scope: storage
  properties: {
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'ba92f5b4-2d11-453d-a403-e96b0029c9fe')
  }
}
resource dns 'Microsoft.Network/privateDnsZones@2020-06-01' = {
  name: '${prefix}.postgres.database.azure.com'
  location: 'global'
  tags: tags
}
resource dnsLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2020-06-01' = {
  parent: dns
  name: 'lab'
  location: 'global'
  properties: { registrationEnabled: false, virtualNetwork: { id: network.outputs.networkId } }
}
resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: '${prefix}-logs'
  location: location
  tags: tags
  properties: { retentionInDays: 30, sku: { name: 'PerGB2018' } }
}
resource environment 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: '${prefix}-environment'
  location: location
  tags: tags
  properties: {
    vnetConfiguration: { infrastructureSubnetId: network.outputs.coordinatorSubnetId, internal: false }
    workloadProfiles: [{ name: 'Consumption', workloadProfileType: 'Consumption' }]
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: { customerId: logs.properties.customerId, sharedKey: logs.listKeys().primarySharedKey }
    }
  }
}
output registry string = registry.properties.loginServer
output identityId string = identity.id
output identityClientId string = identity.properties.clientId
output environmentId string = environment.id
output origin string = 'https://${prefix}.${environment.properties.defaultDomain}'
output workerSubnetId string = network.outputs.subnetId
