type ContentPart = image | string

class ChatMessage {
  role "user" | "assistant"
  content ContentPart[]
}

template_string RenderContent(parts: ContentPart[]) #"
    {% for part in parts -%}{{part}}{%- endfor %}
"#

template_string RenderUserContent(parts: ContentPart[], use_cache: bool) #"
    {% for part in parts -%}
    {% if use_cache and loop.last -%}
    {{ _.role("user", cache_control={"type": "ephemeral"}) }}
    {%- else -%}
    {{ _.role("user") }}
    {%- endif -%}
    {{part}}
    {%- endfor %}
"#

template_string RenderAssistantContent(parts: ContentPart[], use_cache: bool) #"
    {% for part in parts -%}
    {% if use_cache and loop.last -%}
    {{ _.role("assistant", cache_control={"type": "ephemeral"}) }}
    {%- else -%}
    {{ _.role("assistant") }}
    {%- endif -%}
    {{part}}
    {%- endfor %}
"#