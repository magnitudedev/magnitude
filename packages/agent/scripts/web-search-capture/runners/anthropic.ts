import type { SearchAuth, SearchOptions } from "../../../src/tools/web-search";
import { anthropicWebSearch } from "../../../src/tools/web-search-anthropic";

export function runAnthropicDirectAdapter(query: string, auth: SearchAuth, options?: SearchOptions) {
  return anthropicWebSearch(query, auth, options);
}
