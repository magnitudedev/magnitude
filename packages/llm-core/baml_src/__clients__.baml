// Learn more about clients at https://docs.boundaryml.com/docs/snippets/clients/overview

client<llm> BedrockSonnet {
  provider aws-bedrock
  options {
    model "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    inference_configuration {
      max_tokens 8192
      temperature 0.7
    }
    additional_model_request_fields {
      stop_sequences ["</function_calls>"]
    }
    allowed_role_metadata ["cache_control"]
  }
}

client<llm> BedrockSonnetChat {
  provider aws-bedrock
  options {
    model "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    inference_configuration {
      max_tokens 8192
      temperature 0.7
    }
    additional_model_request_fields {
      stop_sequences ["</function_calls>"]
    }
    allowed_role_metadata ["cache_control"]
  }
}

client<llm> LocalQwen {
  provider openai-generic
  options {
    base_url "http://localhost:1234/v1"
    model "qwen/qwen3-32b"
    temperature 1.0
  }
}

client<llm> CerebrasGLM {
  provider openai-generic
  options {
    base_url "https://api.cerebras.ai/v1"
    model "zai-glm-4.7"
    api_key env.CEREBRAS_API_KEY
    temperature 1.0
  }
}

client<llm> GrokFast {
  provider openai-generic
  options {
    base_url "https://api.x.ai/v1"
    model "grok-4-1-fast-non-reasoning"
    api_key env.XAI_API_KEY
    temperature 1.0
  }
}

function StopSeqTest() -> string {
  client ChatClientNoRetry
  prompt #"
    Respond with "Hello" then "</function_calls>" then "Goodbye"
  "#
}

function GeminiSearchControlTest() -> string {
  client VertexGeminiFlash
  prompt #"
    Find an event where this person was, in person:

    Company: Triumvirate Environmental

    Email: jmcquillan@triumvirate.com 
  "#
}

function GeminiSearchTest() -> string {
  client VertexGeminiFlashWithSearch
  prompt #"
    Find an event where this person was, in person:

    Company: Triumvirate Environmental

    Email: jmcquillan@triumvirate.com 
  "#
}

function GeminiPro3Test() -> string {
  client VertexGemini3Pro
  prompt #"
    Tell me a haiku about sloths
  "#
}

client<llm> OrchestratorClient {
  provider fallback
  options {
    strategy [
      VertexGeminiPro,
      GeminiProOR
    ]
  }
}

client<llm> NodeGeneratorClient {
  provider fallback
  options {
    strategy [
      VertexGeminiFlash,
      GeminiFlashOR
    ]
  }
}

client<llm> MarkdownGeneratorClient {
  provider fallback
  options {
    strategy [
      VertexGeminiFlashLite,
      GeminiFlashLiteOR
    ]
  }
}

client<llm> GenericAIConnectorClient {
  provider fallback
  options {
    strategy [
      VertexGeminiFlash,
      GeminiFlashOR
    ]
  }
}

client<llm> SimulationClient {
  provider fallback
  options {
    strategy [
      VertexGeminiFlashLite,
      GeminiFlashLiteOR
    ]
  }
}

// === Anthropic ===
client<llm> AnthropicSonnet {
  provider anthropic
  retry_policy SimpleConstant
  options {
    model "claude-sonnet-4-5-20250929"
    api_key env.ANTHROPIC_API_KEY
    max_tokens 8192
    temperature 0.7
    stop_sequences ["</function_calls>"]
    allowed_role_metadata ["cache_control"]
  }
}

// Chat-specific client without retry - retries handled at call site for better timeout control
client<llm> AnthropicSonnetChat {
  provider anthropic
  options {
    model "claude-sonnet-4-5-20250929"
    api_key env.ANTHROPIC_API_KEY
    max_tokens 8192
    temperature 0.7
    stop_sequences ["</function_calls>"]
    allowed_role_metadata ["cache_control"]
  }
}

client<llm> AnthropicHaiku {
  provider anthropic
  retry_policy SimpleConstant
  options {
    model "claude-haiku-4-5-20251001"
    api_key env.ANTHROPIC_API_KEY
    max_tokens 200
    temperature 1.0
    allowed_role_metadata ["cache_control"]
  }
}


client<llm> OpenAIGPT5Mini {
  provider openai
  options {
    api_key env.OPENAI_API_KEY
    model "gpt-5-mini-2025-08-07"
  }
}

client<llm> VertexGemini3Flash {
  provider vertex-ai
  retry_policy SimpleConstant
  options {
    model "gemini-3-flash-preview"
    location global
    credentials env.GOOGLE_APPLICATION_CREDENTIALS
    generationConfig {
      temperature 1.0
      thinkingConfig {
        thinkingLevel MINIMAL
      }
    }
  }
}

client<llm> VertexGemini3Pro {
  provider vertex-ai
  retry_policy SimpleConstant
  options {
    model "gemini-3-pro-preview"
    location global
    credentials env.GOOGLE_APPLICATION_CREDENTIALS
    generationConfig {
      temperature 1.0
      thinkingConfig {
        thinkingLevel LOW
      }
    }
  }
}

client<llm> VertexGeminiPro {
  provider vertex-ai
  retry_policy SimpleConstant
  options {
    model "gemini-2.5-pro"
    location us-central1
    credentials env.GOOGLE_APPLICATION_CREDENTIALS
    generationConfig {
      temperature 0.7
      thinkingConfig {
        thinkingBudget 512
      }
    }
  }
}

client<llm> VertexGeminiFlash {
  provider vertex-ai
  retry_policy SimpleConstant
  options {
    model "gemini-2.5-flash"
    location us-central1
    credentials env.GOOGLE_APPLICATION_CREDENTIALS
    generationConfig {
      temperature 0.5
      thinkingConfig {
        thinkingBudget 0
      }
    }
  }
}

// With search grounded + adapting thinking budget (it can call search in the adaptive thinking window)
client<llm> VertexGeminiFlashWithSearch {
  provider vertex-ai
  retry_policy SimpleConstant
  options {
    model "gemini-2.5-flash"
    location us-central1
    credentials env.GOOGLE_APPLICATION_CREDENTIALS
    generationConfig {
      temperature 0.5
      thinkingConfig {
        thinkingBudget -1
      }
    }
    tools [
      {
        googleSearch {}
      }
    ]
  }
}

client<llm> VertexGeminiFlashThinking {
  provider vertex-ai
  retry_policy SimpleConstant
  options {
    model "gemini-2.5-flash"
    location us-central1
    credentials env.GOOGLE_APPLICATION_CREDENTIALS
    generationConfig {
      temperature 0.7
      thinkingConfig {
        thinkingBudget 512
      }
    }
  }
}

client<llm> VertexGeminiFlashLite {
  provider vertex-ai
  retry_policy SimpleConstant
  options {
    model "gemini-2.5-flash-lite"
    location us-central1
    credentials env.GOOGLE_APPLICATION_CREDENTIALS
    generationConfig {
      temperature 0.5
      //thinkingBudget 0
    }
  }
}

client<llm> GeminiPro {
  provider google-ai
  retry_policy SimpleConstant
  options {
    model "gemini-2.5-pro"
    generationConfig {
      temperature 0.5
      thinkingConfig {
        thinkingBudget 128
      }
    }
  }
}

client<llm> GeminiFlash {
  provider google-ai
  retry_policy SimpleConstant
  options {
    model "gemini-2.5-flash"
    generationConfig {
      temperature 0.5
      thinkingConfig {
        thinkingBudget 0
      }
    }
  }
}

client<llm> GeminiFlashLite {
  provider google-ai
  retry_policy SimpleConstant
  options {
    model "gemini-2.5-flash-lite"
    generationConfig {
      temperature 0.5
      //thinkingBudget 0
    }
  }
}

client<llm> GeminiProOR {
  provider openai-generic
  retry_policy SimpleConstant
  options {
    base_url "https://openrouter.ai/api/v1"
    api_key env.OPENROUTER_API_KEY
    model "google/gemini-2.5-pro"
    generationConfig {
      temperature 0.5
      thinkingConfig {
        thinkingBudget 128
      }
    }
  }
}

client<llm> GeminiFlashOR {
  provider openai-generic
  retry_policy SimpleConstant
  options {
    base_url "https://openrouter.ai/api/v1"
    api_key env.OPENROUTER_API_KEY
    model "google/gemini-2.5-flash"
    generationConfig {
      temperature 0.5
      thinkingConfig {
        thinkingBudget 0
      }
    }
  }
}

client<llm> GeminiFlashLiteOR {
  provider openai-generic
  retry_policy SimpleConstant
  options {
    base_url "https://openrouter.ai/api/v1"
    api_key env.OPENROUTER_API_KEY
    model "google/gemini-2.5-flash-lite"
    generationConfig {
      temperature 0.5
    }
  }
}


client<llm> Qwen {
  provider openai-generic
  retry_policy SimpleConstant
  options {
    base_url "https://openrouter.ai/api/v1"
    model "qwen/qwen2.5-vl-72b-instruct"
    api_key env.OPENROUTER_API_KEY
    temperature 0.0
  }
}

retry_policy SimpleConstant {
  max_retries 3
  strategy {
    type constant_delay
    delay_ms 200
  }
}

retry_policy SlowExp {
  max_retries 5
  strategy {
    type exponential_backoff
    delay_ms 200
    multiplier 2
    max_delay_ms 6400
  }
}

// Anthropic Sonnet with Web Search
client<llm> AnthropicSonnetWithSearch {
  provider anthropic
  options {
    model "claude-sonnet-4-20250514"
    api_key env.ANTHROPIC_API_KEY
    max_tokens 4096
    temperature 0.7
    headers {
      "anthropic-beta" "web-search-2025-03-05"
    }
    tools [
      {
        type "web_search_20250305"
        name "web_search"
        max_uses 8
      }
    ]
  }
}
