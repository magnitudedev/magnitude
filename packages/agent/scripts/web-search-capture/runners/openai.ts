import type { SearchAuth, SearchOptions } from "../../../src/tools/web-search";
import { openaiWebSearch } from "../../../src/tools/web-search-openai";

export function runOpenAIDirectAdapter(query: string, auth: SearchAuth, options?: SearchOptions) {
  return openaiWebSearch(query, auth, options);
}
