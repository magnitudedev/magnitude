import type { SearchAuth, SearchOptions } from "../../../src/tools/web-search";
import { copilotWebSearch } from "../../../src/tools/web-search-copilot";

export function runCopilotDirectAdapter(query: string, auth: SearchAuth, options?: SearchOptions) {
  return copilotWebSearch(query, auth, options);
}
