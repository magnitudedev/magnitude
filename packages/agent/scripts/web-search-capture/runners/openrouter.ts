import type { SearchAuth, SearchOptions } from "../../../src/tools/web-search";
import { openrouterWebSearch } from "../../../src/tools/web-search-openrouter";

export function runOpenRouterDirectAdapter(query: string, auth: SearchAuth, options?: SearchOptions) {
  return openrouterWebSearch(query, auth, options);
}
