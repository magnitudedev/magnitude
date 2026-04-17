/**
 * Global Tools
 *
 * Built-in tools available in the sandbox:
 * - webFetch(url) - Fetch and extract web page content
 */

// Web search tool temporarily disabled — awaiting Exa-based reimplementation
// import { webSearchTool, webSearchXmlBinding } from './web-search-tool'
import { webFetchTool, webFetchXmlBinding } from './web-fetch-tool'

// =============================================================================
// Global Tools
// =============================================================================

export const globalTools = [webFetchTool]

export const globalXmlBindings = [webFetchXmlBinding]
