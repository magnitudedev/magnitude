/**
 * Web Search
 *
 * Provider-specific web search shims have been removed.
 * Web search will be reimplemented using Exa (via Magnitude provider or user-provided API key).
 *
 * This module is kept as a skeleton for the future implementation,
 * preserving useful type definitions and the tool definition.
 */

import { Effect } from 'effect'
import { Schema } from '@effect/schema'
import { defineTool, ToolErrorSchema } from '@magnitudedev/tools'
import { defineXmlBinding } from '@magnitudedev/xml-act'

// =============================================================================
// Shared Types
// =============================================================================

export interface WebSearchResult {
  title: string;
  url: string;
}

export interface WebSearchToolResult {
  tool_use_id: string;
  content: WebSearchResult[];
}

export interface WebSearchResponse {
  query: string;
  results: (WebSearchToolResult | string)[];
  textResponse: string;
  usage: {
    input_tokens: number;
    output_tokens: number;
    web_search_requests: number;
  };
}

export interface SearchAuth {
  type: "api-key";
  value: string;
}

export interface SearchOptions {
  system?: string;
  allowed_domains?: string[];
  blocked_domains?: string[];
  model?: string;
  max_tokens?: number;
}

// =============================================================================
// Tool Definition
// =============================================================================

const WebSearchErrorSchema = ToolErrorSchema('WebSearchError', {})

export const webSearchTool = defineTool({
  name: 'web-search',
  group: 'default',
  description: 'Search the web and optionally extract structured data',

  inputSchema: Schema.Struct({
    query: Schema.String,
    schema: Schema.optional(Schema.Record({ key: Schema.String, value: Schema.Unknown }))
  }),

  outputSchema: Schema.Struct({
    text: Schema.String,
    sources: Schema.Array(Schema.Struct({ title: Schema.String, url: Schema.String })),
    data: Schema.optional(Schema.Unknown),
  }),
  errorSchema: WebSearchErrorSchema,

  execute: ({ query }, _ctx) =>
    Effect.fail({ _tag: 'WebSearchError' as const, message: 'Web search is not yet reimplemented with Exa. Coming soon.' }),

  label: (input) => input.query ? `Searching: ${input.query.slice(0, 50)}` : 'Searching…',
})

export const webSearchXmlBinding = defineXmlBinding(webSearchTool, {
  input: { body: 'query' },
  output: {
    body: 'text',
    children: [{
      field: 'sources',
      tag: 'source',
      attributes: [{ field: 'title', attr: 'title' }, { field: 'url', attr: 'url' }],
    }],
  },
} as const)
