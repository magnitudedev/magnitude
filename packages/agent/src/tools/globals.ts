/**
 * Global Tools
 *
 * Built-in tools available in the sandbox:
 * - webSearch(query, schema?) - Search the web with optional structured output
 */

import { webSearchTool, webSearchXmlBinding } from './web-search-tool'
import { webFetchTool, webFetchXmlBinding } from './web-fetch-tool'

// =============================================================================
// Global Tools
// =============================================================================

export const globalTools = [webSearchTool, webFetchTool]

export const globalXmlBindings = [webSearchXmlBinding, webFetchXmlBinding]
