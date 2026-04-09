import type { SearchAuth, SearchOptions } from "../../../src/tools/web-search";
import { vercelWebSearch } from "../../../src/tools/web-search-vercel";

export function runVercelDirectAdapter(query: string, auth: SearchAuth, options?: SearchOptions) {
  return vercelWebSearch(query, auth, options);
}
