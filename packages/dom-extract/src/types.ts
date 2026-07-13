/**
 * Core types for the unstructured-ts library
 * Comprehensive element system matching Unstructured Python library capabilities
 */

export enum ElementType {
  // Basic text elements
  TITLE = 'Title',
  NARRATIVE_TEXT = 'NarrativeText',
  TEXT = 'Text',
  UNCATEGORIZED_TEXT = 'UncategorizedText',
  BULLETED_TEXT = 'BulletedText',
  PARAGRAPH = 'Paragraph',
  
  // Specialized text elements
  ABSTRACT = 'Abstract',
  CAPTION = 'Caption',
  FIGURE_CAPTION = 'FigureCaption',
  FOOTNOTE = 'Footnote',
  PAGE_NUMBER = 'PageNumber',
  
  // List elements
  LIST = 'List',
  LIST_ITEM = 'ListItem',
  
  // Structural elements
  HEADER = 'Header',
  FOOTER = 'Footer',
  PAGE_HEADER = 'PageHeader',
  PAGE_FOOTER = 'PageFooter',
  SECTION_HEADER = 'SectionHeader',
  HEADLINE = 'Headline',
  SUB_HEADLINE = 'Subheadline',
  PAGE_BREAK = 'PageBreak',
  
  // Media elements
  IMAGE = 'Image',
  PICTURE = 'Picture',
  FIGURE = 'Figure',
  
  // Table elements
  TABLE = 'Table',
  
  // Form elements
  FORM = 'Form',
  FIELD_NAME = 'FieldName',
  VALUE = 'Value',
  FORM_KEYS_VALUES = 'FormKeysValues',
  CHECK_BOX_CHECKED = 'CheckBoxChecked',
  CHECK_BOX_UNCHECKED = 'CheckBoxUnchecked',
  RADIO_BUTTON_CHECKED = 'RadioButtonChecked',
  RADIO_BUTTON_UNCHECKED = 'RadioButtonUnchecked',
  
  // Code elements
  CODE_SNIPPET = 'CodeSnippet',
  FORMULA = 'Formula',
  
  // Contact elements (removed - caused false positives)
  
  // Link elements
  LINK = 'Link',
  
  // Navigation elements (usually filtered)
  NAVIGATION = 'Navigation',
  
  // Composite elements
  COMPOSITE_ELEMENT = 'CompositeElement',
  
  // Document metadata
  DOCUMENT_DATA = 'DocumentData'
}

// Coordinate system for layout information
export interface Point {
  x: number;
  y: number;
}

export interface CoordinateSystem {
  width: number;
  height: number;
  coordinateUnit: 'pixels' | 'points' | 'inches';
}

export interface CoordinatesMetadata {
  points?: Point[];
  system?: CoordinateSystem;
  layoutWidth?: number;
  layoutHeight?: number;
}

// Data source metadata
export interface DataSourceMetadata {
  url?: string;
  version?: string;
  recordLocator?: Record<string, any>;
  dateCreated?: string;
  dateModified?: string;
  dateProcessed?: string;
  permissionsData?: Array<Record<string, any>>;
}

// Link metadata
export interface Link {
  text: string;
  url: string;
  startIndex?: number;
}

// Form field metadata
export interface FormField {
  fieldName: string;
  fieldValue: string;
  fieldType?: string;
}

// Comprehensive element metadata matching Unstructured Python library
export interface ElementMetadata {
  // File and document metadata
  filename?: string;
  filetype?: string;
  fileDirectory?: string;
  lastModified?: string;
  
  // Page and layout metadata
  pageNumber?: number;
  pageName?: string;
  coordinates?: CoordinatesMetadata;
  
  // Hierarchy and relationships
  parentId?: string;
  categoryDepth?: number;
  
  // DOM-specific metadata
  tagName?: string;
  cssClasses?: string[];
  elementId?: string;
  
  // Content analysis metadata
  textLength?: number;
  detectionClassProb?: number;
  isContinuation?: boolean;
  
  // Link metadata
  linkTexts?: string[];
  linkUrls?: string[];
  linkStartIndexes?: number[];
  links?: Link[];
  
  // Emphasis and formatting
  emphasizedTextContents?: string[];
  emphasizedTextTags?: string[];
  
  // Email metadata
  emailMessageId?: string;
  sentFrom?: string[];
  sentTo?: string[];
  ccRecipient?: string[];
  bccRecipient?: string[];
  subject?: string;
  signature?: string;
  
  // Form metadata
  keyValuePairs?: FormField[];
  
  // Image metadata
  imageBase64?: string;
  imageMimeType?: string;
  imageUrl?: string;
  imagePath?: string;
  
  // Table metadata
  textAsHtml?: string;
  tableAsCells?: Record<string, string | number>;
  
  // Data source metadata
  dataSource?: DataSourceMetadata;
  
  // Language metadata
  languages?: string[];
  
  // Header/footer classification
  headerFooterType?: string;
  
  // Original elements for composite elements
  origElements?: Element[];
  
  // Debug and processing metadata
  detectionOrigin?: string;
  url?: string;
  attachedToFilename?: string;
  
  // Raw HTML for debugging/advanced use
  originalHtml?: string;
}

// Base element interface
export interface Element {
  id: string;
  type: ElementType;
  text: string;
  metadata: ElementMetadata;
}

// Specialized element interfaces
export interface TableElement extends Element {
  type: ElementType.TABLE;
  rows: string[][];
  headers?: string[];
}

export interface ImageElement extends Element {
  type: ElementType.IMAGE;
  src?: string;
  alt?: string;
  width?: number;
  height?: number;
}

export interface FormElement extends Element {
  type: ElementType.FORM | ElementType.FORM_KEYS_VALUES;
  fields?: FormField[];
}

export interface CheckBoxElement extends Element {
  type: ElementType.CHECK_BOX_CHECKED | ElementType.CHECK_BOX_UNCHECKED;
  checked: boolean;
  value?: string;
}

export interface RadioButtonElement extends Element {
  type: ElementType.RADIO_BUTTON_CHECKED | ElementType.RADIO_BUTTON_UNCHECKED;
  checked: boolean;
  value?: string;
  groupName?: string;
}

export interface LinkElement extends Element {
  type: ElementType.LINK;
  url: string;
  linkText: string;
}

// AddressElement and EmailAddressElement removed - caused false positives

export interface CodeElement extends Element {
  type: ElementType.CODE_SNIPPET;
  language?: string;
  codeBlock: string;
}

export interface FormulaElement extends Element {
  type: ElementType.FORMULA;
  formula: string;
  formulaType?: 'latex' | 'mathml' | 'text';
}

export interface CompositeElement extends Element {
  type: ElementType.COMPOSITE_ELEMENT;
  elements: Element[];
}

// Processing strategy options
export enum ProcessingStrategy {
  AUTO = 'auto',
  FAST = 'fast',
  ACCURATE = 'accurate',
  OCR_ONLY = 'ocr_only'
}

// Chunking strategy options
export enum ChunkingStrategy {
  NONE = 'none',
  BASIC = 'basic',
  BY_TITLE = 'by_title',
  BY_PAGE = 'by_page',
  BY_SIMILARITY = 'by_similarity'
}

// Comprehensive partition options
export interface PartitionOptions {
  // Content filtering options
  skipNavigation?: boolean;
  skipHeaders?: boolean;
  skipFooters?: boolean;
  skipForms?: boolean;
  skipHeadersAndFooters?: boolean;
  
  // Text processing options
  minTextLength?: number;
  maxTextLength?: number;
  preserveWhitespace?: boolean;
  
  // Table processing options
  extractTables?: boolean;
  inferTableStructure?: boolean;
  skipInferTableTypes?: string[];
  
  // Image processing options
  extractImages?: boolean;
  includeImageAlt?: boolean;
  extractImageBlockTypes?: string[];
  extractImageBlockToPayload?: boolean;
  extractImageBlockOutputDir?: string;
  
  // Form processing options
  extractForms?: boolean;
  extractFormFields?: boolean;
  
  // Link processing options
  extractLinks?: boolean;
  
  // Language processing options
  languages?: string[];
  detectLanguagePerElement?: boolean;
  
  // Coordinate and layout options
  includeCoordinates?: boolean;
  coordinateSystem?: CoordinateSystem;
  
  // Hierarchy options
  includePageBreaks?: boolean;
  maintainHierarchy?: boolean;
  
  // Processing strategy
  strategy?: ProcessingStrategy;
  
  // Chunking options
  chunkingStrategy?: ChunkingStrategy;
  maxCharacters?: number;
  newAfterNChars?: number;
  combineTextUnderNChars?: number;
  
  // Metadata options
  includeOriginalHtml?: boolean;
  includeMetadata?: boolean;
  metadataFilename?: string;
  uniqueElementIds?: boolean;
  
  // Email-specific options
  processAttachments?: boolean;
  attachmentPartitioningStrategy?: ProcessingStrategy;
  
  // Advanced filtering
  elementTypeFilters?: ElementType[];
  contentFilters?: {
    minWords?: number;
    maxWords?: number;
    excludePatterns?: RegExp[];
    includePatterns?: RegExp[];
  };
  
  // Debug and development options
  includeDebugMetadata?: boolean;
  detectionOrigin?: string;
}

// Comprehensive partition result
export interface PartitionResult {
  elements: Element[];
  metadata: {
    totalElements: number;
    processingTime?: number;
    warnings?: string[];
    errors?: string[];
    
    // Document-level metadata
    filename?: string;
    filetype?: string;
    pageCount?: number;
    
    // Processing statistics
    elementTypeCounts?: Record<ElementType, number>;
    averageElementLength?: number;
    
    // Language detection results
    detectedLanguages?: string[];
    
    // Extraction statistics
    tablesExtracted?: number;
    imagesExtracted?: number;
    formsExtracted?: number;
    linksExtracted?: number;
    
    // Performance metrics
    memoryUsage?: number;
    
    // Data source information
    dataSource?: DataSourceMetadata;
  };
}

// Union type for all element types
export type AnyElement = 
  | Element
  | TableElement
  | ImageElement
  | FormElement
  | CheckBoxElement
  | RadioButtonElement
  | LinkElement
  | CodeElement
  | FormulaElement
  | CompositeElement;