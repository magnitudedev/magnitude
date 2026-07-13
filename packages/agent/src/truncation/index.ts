// Shared budget primitives
export type { Measurement } from './budget'
export { allocateBudget, charsToTokensUpper, charsToTokensLower } from './budget'

// Token estimation
export { DEFAULT_IMAGE_TOKENS, estimateImageTokens, estimateText, estimateContentTokens } from './estimate'

// Formatting
export { formatSize } from './format'

// JSON truncation
export type { JsonValue } from '@magnitudedev/ai'
export { measureBounded } from './json/measure'
export { truncate, truncateMany } from './json/truncate'

// XML body truncation
export { truncateXmlBodyString, renderXmlBodyValue, renderXmlBodyValues } from './xml-body'

// Shape description
export { describeShape } from './describe-shape'

// Folder tree
export type { FolderNode } from './folder-tree/tree'
export { buildFolderTree } from './folder-tree/build'
export { truncateFolderTree } from './folder-tree/truncate'
