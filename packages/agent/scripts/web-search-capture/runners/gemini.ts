import type { SearchAuth, SearchOptions } from "../../../src/tools/web-search";
import { geminiWebSearch } from "../../../src/tools/web-search-gemini";

export function runGeminiDirectAdapter(query: string, auth: SearchAuth, options?: SearchOptions) {
  return geminiWebSearch(query, auth, options);
}
