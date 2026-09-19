targetScope = 'resourceGroup'
param location string = 'westus3'
param prefix string = 'magnitude-lab'
param serverName string = 'magnitude-lab-db-5304'
@secure()
param password string

resource coordinatorNetwork 'Microsoft.Network/virtualNetworks@2024-05-01' existing = { name: '${prefix}-network' }
resource dns 'Microsoft.Network/privateDnsZones@2020-06-01' existing = { name: '${prefix}.postgres.database.azure.com' }
resource databaseNetwork 'Microsoft.Network/virtualNetworks@2024-05-01' = {
  name: '${prefix}-database-network'
  location: location
  tags: { 'lab-component': 'database' }
  properties: {
    addressSpace: { addressPrefixes: ['10.95.0.0/16'] }
    subnets: [{ name: 'database', properties: {
      addressPrefix: '10.95.0.0/24'
      delegations: [{ name: 'postgres', properties: { serviceName: 'Microsoft.DBforPostgreSQL/flexibleServers' } }]
    } }]
  }
}
resource toDatabase 'Microsoft.Network/virtualNetworks/virtualNetworkPeerings@2024-05-01' = {
  parent: coordinatorNetwork
  name: 'database'
  properties: { allowVirtualNetworkAccess: true, allowForwardedTraffic: false, remoteVirtualNetwork: { id: databaseNetwork.id } }
}
resource toCoordinator 'Microsoft.Network/virtualNetworks/virtualNetworkPeerings@2024-05-01' = {
  parent: databaseNetwork
  name: 'coordinator'
  properties: { allowVirtualNetworkAccess: true, allowForwardedTraffic: false, remoteVirtualNetwork: { id: coordinatorNetwork.id } }
}
resource link 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2020-06-01' = {
  parent: dns
  name: 'database'
  location: 'global'
  properties: { registrationEnabled: false, virtualNetwork: { id: databaseNetwork.id } }
}
resource server 'Microsoft.DBforPostgreSQL/flexibleServers@2024-08-01' = {
  name: serverName
  location: location
  tags: { 'lab-component': 'database' }
  sku: { name: 'Standard_B1ms', tier: 'Burstable' }
  properties: {
    version: '16'
    administratorLogin: 'labadmin'
    administratorLoginPassword: password
    storage: { storageSizeGB: 32, autoGrow: 'Enabled' }
    backup: { backupRetentionDays: 7, geoRedundantBackup: 'Disabled' }
    highAvailability: { mode: 'Disabled' }
    network: { delegatedSubnetResourceId: '${databaseNetwork.id}/subnets/database', privateDnsZoneArmResourceId: dns.id, publicNetworkAccess: 'Disabled' }
  }
  dependsOn: [link]
}
resource database 'Microsoft.DBforPostgreSQL/flexibleServers/databases@2024-08-01' = {
  parent: server
  name: 'testinglab'
  properties: { charset: 'UTF8', collation: 'en_US.utf8' }
}
output host string = server.properties.fullyQualifiedDomainName
