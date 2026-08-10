# Custom endpoints

Add an OpenAI-compatible Chat Completions endpoint to `~/.magnitude/config.json`. Declare the connection and the models you want Magnitude to expose, then save the file. Magnitude applies valid changes automatically.

## Example

Add an endpoint under `providers`, preserving any other fields already in the file. This example configures OpenRouter with GLM 5.2:

```json
{
  "providers": {
    "openrouter": {
      "displayName": "OpenRouter",
      "connection": {
        "baseUrl": "https://openrouter.ai/api/v1",
        "authentication": {
          "type": "bearer",
          "credential": {
            "type": "environment",
            "variable": "OPENROUTER_API_KEY"
          }
        }
      },
      "models": {
        "z-ai/glm-5.2": {
          "displayName": "GLM 5.2",
          "contextWindow": 1048576,
          "maxOutputTokens": 128000,
          "capabilities": {
            "reasoning": {
              "efforts": ["high", "xhigh"],
              "defaultEffort": "high"
            }
          }
        }
      }
    }
  }
}
```

Set the referenced credential before starting Magnitude:

```bash
export OPENROUTER_API_KEY="your-key"
```

## Configuration shape

```ts
type CustomEndpointsConfig = {
  providers?: Record<string, { // Stable endpoint key
    displayName: string
    connection: {
      baseUrl: string
      authentication:
        | { type: "none" }
        | { type: "bearer"; credential: EnvironmentCredential }
        | { type: "header"; name: string; credential: EnvironmentCredential }
      headers?: Record<string, string>
    }
    models: Record<string, { // Exact model ID sent to the endpoint
      displayName: string
      contextWindow: number
      maxOutputTokens: number
      capabilities?: {
        vision?: boolean
        reasoning?: {
          efforts: string[]
          defaultEffort: string
        }
      }
    }>
  }>
}

type EnvironmentCredential = {
  type: "environment"
  variable: string
}
```

For any endpoint:

- Use its API root as `baseUrl`; do not include `/chat/completions`.
- Use each model's exact API model ID as its key under `models`.
- Declare the model's actual context and output limits.
- Reference secrets through environment variables instead of storing them in the file.

After you save the file, configured models appear in Magnitude's normal model picker. Removing a selected endpoint or model clears that model slot without choosing a replacement. Restoring the same keys makes the model available again but does not reselect it.
