targetScope = 'resourceGroup'
param location string = 'westus2'
param prefix string = 'magnitude-lab'
param registryName string = 'magnitudelab5304'
@description('Immutable registry image reference including its digest.')
param image string
@description('Unique revision suffix, changed whenever image or mounted configuration changes.')
param revision string
@secure()
param databaseUrl string
@secure()
param coordinatorConfigBase64 string
@secure()
param workerInitializationBase64 string
@secure()
param developerToken string

resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' existing = { name: '${prefix}-coordinator' }
resource environment 'Microsoft.App/managedEnvironments@2024-03-01' existing = { name: '${prefix}-environment' }
resource registry 'Microsoft.ContainerRegistry/registries@2023-07-01' existing = { name: registryName }
resource coordinator 'Microsoft.App/containerApps@2024-03-01' = {
  name: prefix
  location: location
  tags: { 'lab-component': 'coordinator' }
  identity: { type: 'UserAssigned', userAssignedIdentities: { '${identity.id}': {} } }
  properties: {
    managedEnvironmentId: environment.id
    workloadProfileName: 'Consumption'
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: { external: true, targetPort: 8080, transport: 'http', allowInsecure: false }
      registries: [{ server: registry.properties.loginServer, identity: identity.id }]
      secrets: [
        { name: 'database-url', value: databaseUrl }
        { name: 'coordinator-config', value: coordinatorConfigBase64 }
        { name: 'worker-initialization', value: workerInitializationBase64 }
        { name: 'developer-token', value: developerToken }
      ]
    }
    template: {
      revisionSuffix: revision
      scale: { minReplicas: 1, maxReplicas: 1 }
      terminationGracePeriodSeconds: 600
      containers: [{
        name: 'coordinator'
        image: image
        resources: { cpu: 2, memory: '4Gi' }
        env: [
          { name: 'LAB_AZURE_CLIENT_ID', value: identity.properties.clientId }
          { name: 'LAB_DATABASE_URL', secretRef: 'database-url' }
          { name: 'LAB_COORDINATOR_CONFIG_BASE64', secretRef: 'coordinator-config' }
          { name: 'LAB_WORKER_INITIALIZATION_BASE64', secretRef: 'worker-initialization' }
          { name: 'LAB_DEVELOPER_TOKEN', secretRef: 'developer-token' }
        ]
        probes: [
          { type: 'Startup', tcpSocket: { port: 8080 }, periodSeconds: 5, failureThreshold: 60 }
          { type: 'Liveness', tcpSocket: { port: 8080 }, periodSeconds: 30, failureThreshold: 3 }
          { type: 'Readiness', tcpSocket: { port: 8080 }, periodSeconds: 10, failureThreshold: 3 }
        ]
      }]
    }
  }
}
output origin string = 'https://${coordinator.properties.configuration.ingress.fqdn}'
