// Azure Infrastructure for Advanced Agentic RAG API
// Deploys: Log Analytics, App Insights, ACR, Key Vault, ACA Environment, ACA App

@description('Location for all resources')
param location string = resourceGroup().location

@description('Base name for resources (lowercase, alphanumeric)')
param baseName string = 'agenticrag'

@description('Container image tag')
param imageTag string = 'latest'

@description('OpenAI API Key (stored in Key Vault)')
@secure()
param openaiApiKey string = ''

@description('Vectara API Key for HHEM hallucination detection (Vectara backend)')
@secure()
param vectaraApiKey string = ''

@description('Vectara Customer ID for HHEM hallucination detection (Vectara backend)')
param vectaraCustomerId string = ''

@description('HHEM backend: local (HuggingFace model) or vectara (managed API)')
@allowed(['local', 'vectara'])
param hhemBackend string = 'vectara'

@description('Model tier for the RAG system')
@allowed(['budget', 'balanced', 'premium'])
param modelTier string = 'budget'

@description('Enable Redis semantic cache for query-level caching')
param cacheEnabled bool = false

@description('Cache similarity threshold (0-1)')
param cacheSimilarityThreshold string = '0.95'

@description('Corpus version for cache namespacing')
param corpusVersion string = 'v1'

// Resource naming
var uniqueSuffix = uniqueString(resourceGroup().id)
var acrName = '${baseName}${uniqueSuffix}'
var keyVaultName = 'kv-${baseName}-${take(uniqueSuffix, 8)}'
var logAnalyticsName = 'log-${baseName}'
var appInsightsName = 'appi-${baseName}'
var acaEnvName = 'acaenv-${baseName}'
var acaAppName = 'aca-${baseName}-api'

// Log Analytics Workspace (required by ACA)
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: logAnalyticsName
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

// Application Insights
resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: appInsightsName
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalytics.id
  }
}

// Azure Container Registry (Basic tier - cost effective for portfolio)
resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: acrName
  location: location
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: true
  }
}

// Key Vault for secrets
resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: keyVaultName
  location: location
  properties: {
    sku: {
      family: 'A'
      name: 'standard'
    }
    tenantId: subscription().tenantId
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 7
  }
}

// Store OpenAI API Key in Key Vault (if provided)
resource openaiSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = if (!empty(openaiApiKey)) {
  parent: keyVault
  name: 'OPENAI-API-KEY'
  properties: {
    value: openaiApiKey
  }
}

// Container Apps Environment
resource acaEnvironment 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: acaEnvName
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
    daprAIInstrumentationKey: appInsights.properties.InstrumentationKey
  }
}

// Container App - RAG API
resource acaApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: acaAppName
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    managedEnvironmentId: acaEnvironment.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8000
        transport: 'http'
        allowInsecure: false
      }
      registries: [
        {
          server: acr.properties.loginServer
          username: acr.listCredentials().username
          passwordSecretRef: 'acr-password'
        }
      ]
      secrets: [
        {
          name: 'acr-password'
          value: acr.listCredentials().passwords[0].value
        }
        {
          name: 'openai-api-key'
          value: !empty(openaiApiKey) ? openaiApiKey : 'placeholder-replace-me'
        }
        {
          name: 'vectara-api-key'
          value: !empty(vectaraApiKey) ? vectaraApiKey : 'placeholder-replace-me'
        }
        {
          name: 'redis-connection-string'
          value: ''  // Set by CI/CD pipeline (Upstash Redis URL)
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'rag-api'
          image: '${acr.properties.loginServer}/${baseName}:${imageTag}'
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
          env: [
            {
              name: 'OPENAI_API_KEY'
              secretRef: 'openai-api-key'
            }
            {
              name: 'VECTARA_API_KEY'
              secretRef: 'vectara-api-key'
            }
            {
              name: 'VECTARA_CUSTOMER_ID'
              value: vectaraCustomerId
            }
            {
              name: 'HHEM_BACKEND'
              value: hhemBackend
            }
            {
              name: 'MODEL_TIER'
              value: modelTier
            }
            {
              name: 'VECTOR_STORE_BACKEND'
              value: 'faiss'
            }
            {
              name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
              value: appInsights.properties.ConnectionString
            }
            {
              name: 'CACHE_ENABLED'
              value: string(cacheEnabled)
            }
            {
              name: 'REDIS_URL'
              secretRef: 'redis-connection-string'
            }
            {
              name: 'CACHE_SIMILARITY_THRESHOLD'
              value: cacheSimilarityThreshold
            }
            {
              name: 'CORPUS_VERSION'
              value: corpusVersion
            }
          ]
          probes: [
            {
              type: 'Liveness'
              httpGet: {
                path: '/v1/health'
                port: 8000
              }
              initialDelaySeconds: 30
              periodSeconds: 30
            }
            {
              type: 'Readiness'
              httpGet: {
                path: '/v1/ready'
                port: 8000
              }
              initialDelaySeconds: 90  // Increased for HHEM local model warmup (default: vectara needs 30s)
              periodSeconds: 10
            }
          ]
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 3
        rules: [
          {
            name: 'http-scaling'
            http: {
              metadata: {
                concurrentRequests: '10'
              }
            }
          }
        ]
      }
    }
  }
}

// Role assignment: ACA can pull from ACR
resource acrPullRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(acr.id, acaApp.id, 'acrpull')
  scope: acr
  properties: {
    principalId: acaApp.identity.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d') // AcrPull
  }
}

// Role assignment: ACA can read Key Vault secrets
resource kvSecretsRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(keyVault.id, acaApp.id, 'kvsecrets')
  scope: keyVault
  properties: {
    principalId: acaApp.identity.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '4633458b-17de-408a-b874-0445c86b69e6') // Key Vault Secrets User
  }
}

// Outputs
output acrLoginServer string = acr.properties.loginServer
output acrName string = acr.name
output acaAppUrl string = 'https://${acaApp.properties.configuration.ingress.fqdn}'
output acaAppName string = acaApp.name
output keyVaultName string = keyVault.name
output appInsightsName string = appInsights.name
output resourceGroupName string = resourceGroup().name
