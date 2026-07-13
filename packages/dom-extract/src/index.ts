/**
 * dom-extract - HTML to markdown extraction
 */

import { DOMPartitioner } from './partitioner';
import { serializeToMarkdown } from './markdown-serializer';
import type { PartitionOptions } from './types';

export interface ExtractOptions {
  minTextLength?: number;
}

const PARTITION_OPTIONS: PartitionOptions = {
  extractImages: true,
  extractForms: true,
  extractLinks: true,
  skipNavigation: false,
  minTextLength: 3,
  includeOriginalHtml: false,
  includeMetadata: true,
};

const SERIALIZER_OPTIONS = {
  includeMetadata: false,
  includePageNumbers: true,
  includeElementIds: false,
  includeCoordinates: false,
  preserveHierarchy: true,
  escapeSpecialChars: true,
  includeFormFields: true,
  includeImageMetadata: true,
};

export function extractHtml(html: string, options?: ExtractOptions): string {
  const partitionOpts: PartitionOptions = {
    ...PARTITION_OPTIONS,
    ...(options?.minTextLength != null ? { minTextLength: options.minTextLength } : {}),
  };
  const partitioner = new DOMPartitioner(partitionOpts);
  const result = partitioner.partition(html);
  return serializeToMarkdown(result, SERIALIZER_OPTIONS);
}

// Re-export internals for advanced usage
export { DOMPartitioner } from './partitioner';
export { DOMCleaner } from './cleaner';
export { ElementClassifier } from './classifier';
export { MarkdownSerializer, serializeToMarkdown } from './markdown-serializer';
export { ElementType } from './types';
export type { PartitionResult, Element, PartitionOptions } from './types';