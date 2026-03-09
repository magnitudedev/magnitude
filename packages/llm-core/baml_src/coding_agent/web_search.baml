// Web Search Function

class WebSearchData {
    @@dynamic
}

function WebSearch(query: string) -> WebSearchData {
  client AnthropicSonnetWithSearch
  prompt #"
    {{ _.role("user") }}
    {{ query }}
    {{ ctx.output_format }}
  "#
}
