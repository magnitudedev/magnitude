targetScope = 'resourceGroup'

@description('Stable infrastructure prefix; worker VM names remain lease-specific.')
param prefix string = 'magnitude-lab'
param location string = resourceGroup().location
param tags object = {
  'lab-component': 'worker-network'
}

resource egressIp 'Microsoft.Network/publicIPAddresses@2024-05-01' = {
  name: '${prefix}-egress-ip'
  location: location
  tags: tags
  sku: { name: 'Standard' }
  properties: {
    publicIPAllocationMethod: 'Static'
  }
}

resource egress 'Microsoft.Network/natGateways@2024-05-01' = {
  name: '${prefix}-egress'
  location: location
  tags: tags
  sku: { name: 'Standard' }
  properties: {
    idleTimeoutInMinutes: 4
    publicIpAddresses: [{ id: egressIp.id }]
  }
}

resource network 'Microsoft.Network/virtualNetworks@2024-05-01' = {
  name: '${prefix}-network'
  location: location
  tags: tags
  properties: {
    addressSpace: { addressPrefixes: ['10.94.0.0/16'] }
    subnets: [{
      name: 'workers'
      properties: {
        addressPrefix: '10.94.1.0/24'
        defaultOutboundAccess: false
        natGateway: { id: egress.id }
      }
    }, {
      name: 'coordinator'
      properties: {
        addressPrefix: '10.94.2.0/23'
        natGateway: { id: egress.id }
        delegations: [{ name: 'container-apps', properties: { serviceName: 'Microsoft.App/environments' } }]
      }
    }]
  }
}

output subnetId string = '${network.id}/subnets/workers'
output coordinatorSubnetId string = '${network.id}/subnets/coordinator'
output networkId string = network.id
