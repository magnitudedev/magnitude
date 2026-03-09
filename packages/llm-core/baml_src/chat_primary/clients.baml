// Chat client WITHOUT retry - for cortex/turn generation where errors are handled gracefully via turn_unexpected_error
client<llm> ChatClientNoRetry {
  provider fallback
  options {
    strategy [
      BedrockSonnetChat,
      AnthropicSonnetChat
    ]
  }
}

client<llm> ChatClientNoRetryAnthropicOnly {
  provider fallback
  options {
    strategy [
      AnthropicSonnetChat
    ]
  }
}

// Chat client WITH retry - for background operations like compaction that need resilience
client<llm> ChatClientWithRetry {
  provider fallback
  options {
    strategy [
      BedrockSonnet,
      AnthropicSonnet
    ]
  }
}

client<llm> ChatClientWithRetryAnthropicOnly {
  provider fallback
  options {
    strategy [
      AnthropicSonnet
    ]
  }
}
