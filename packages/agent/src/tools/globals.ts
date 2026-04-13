/**
 * Global Tools
 *
 * Built-in tools available in the sandbox:
 * - webSearch(query, schema?) - Search the web with optional structured output
 */

import { webSearchTool, webSearchXmlBinding } from './web-search-tool'
import { webFetchTool, webFetchXmlBinding } from './web-fetch-tool'
import { skillTool, skillXmlBinding } from './skill'

// =============================================================================
// Global Tools
// =============================================================================

export { skillTool }

export const globalTools = [webSearchTool, webFetchTool, skillTool]

export const globalXmlBindings = [webSearchXmlBinding, webFetchXmlBinding, skillXmlBinding]
