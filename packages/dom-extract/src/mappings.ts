/**
 * Comprehensive HTML tag and CSS class mappings for element classification
 * Matching Unstructured Python library capabilities
 */

import { ElementType } from './types';

// HTML tag to element type mappings
export const TAG_TO_ELEMENT_TYPE: Record<string, ElementType> = {
  // Headings and titles
  'h1': ElementType.TITLE,
  'h2': ElementType.TITLE,
  'h3': ElementType.TITLE,
  'h4': ElementType.TITLE,
  'h5': ElementType.TITLE,
  'h6': ElementType.TITLE,
  'title': ElementType.TITLE, // Document title
  
  // Text content
  'p': ElementType.NARRATIVE_TEXT,
  'div': ElementType.TEXT, // Will be refined by content analysis
  'span': ElementType.TEXT,
  'article': ElementType.NARRATIVE_TEXT,
  'section': ElementType.NARRATIVE_TEXT,
  'main': ElementType.NARRATIVE_TEXT,
  'aside': ElementType.TEXT,
  
  // Specialized text elements
  'blockquote': ElementType.NARRATIVE_TEXT,
  'q': ElementType.NARRATIVE_TEXT,
  'cite': ElementType.NARRATIVE_TEXT,
  'abbr': ElementType.TEXT,
  'acronym': ElementType.TEXT,
  'dfn': ElementType.TEXT,
  'time': ElementType.TEXT,
  
  // Lists
  'li': ElementType.LIST_ITEM,
  'ul': ElementType.LIST,
  'ol': ElementType.LIST,
  'dl': ElementType.LIST,
  'dt': ElementType.LIST_ITEM,
  'dd': ElementType.LIST_ITEM,
  
  // Tables
  'table': ElementType.TABLE,
  'thead': ElementType.TABLE,
  'tbody': ElementType.TABLE,
  'tfoot': ElementType.TABLE,
  'tr': ElementType.TABLE,
  'td': ElementType.TABLE,
  'th': ElementType.TABLE,
  'caption': ElementType.CAPTION,
  'colgroup': ElementType.TABLE,
  'col': ElementType.TABLE,
  
  // Media elements
  'img': ElementType.IMAGE,
  'figure': ElementType.FIGURE,
  'picture': ElementType.PICTURE,
  'figcaption': ElementType.FIGURE_CAPTION,
  'video': ElementType.IMAGE, // Treat as media element
  'audio': ElementType.IMAGE, // Treat as media element
  'canvas': ElementType.IMAGE,
  'svg': ElementType.IMAGE,
  
  // Code elements
  'code': ElementType.CODE_SNIPPET,
  'pre': ElementType.CODE_SNIPPET,
  'kbd': ElementType.CODE_SNIPPET,
  'samp': ElementType.CODE_SNIPPET,
  'var': ElementType.CODE_SNIPPET,
  
  // Navigation (usually filtered)
  'nav': ElementType.NAVIGATION,
  'menu': ElementType.NAVIGATION,
  'menuitem': ElementType.NAVIGATION,
  
  // Headers/Footers
  'header': ElementType.HEADER,
  'footer': ElementType.FOOTER,
  
  // Forms - only form itself should be mapped directly, others handled by specialized logic
  'form': ElementType.FORM,
  'label': ElementType.FIELD_NAME,
  'legend': ElementType.FIELD_NAME,
  'option': ElementType.VALUE,
  'output': ElementType.VALUE,
  'progress': ElementType.VALUE,
  'meter': ElementType.VALUE,
  
  // Contact information (removed - caused false positives)
  
  // Links
  'a': ElementType.LINK,
  
  // Document structure
  'hr': ElementType.PAGE_BREAK,
  'br': ElementType.TEXT, // Line break, usually part of text
  
  // Mathematical content
  'math': ElementType.FORMULA,
  'mrow': ElementType.FORMULA,
  'mi': ElementType.FORMULA,
  'mn': ElementType.FORMULA,
  'mo': ElementType.FORMULA,
  'mfrac': ElementType.FORMULA,
  'msup': ElementType.FORMULA,
  'msub': ElementType.FORMULA,
  'msubsup': ElementType.FORMULA,
  'munder': ElementType.FORMULA,
  'mover': ElementType.FORMULA,
  'munderover': ElementType.FORMULA,
  'msqrt': ElementType.FORMULA,
  'mroot': ElementType.FORMULA,
  'mtext': ElementType.FORMULA,
  'mspace': ElementType.FORMULA,
  'mstyle': ElementType.FORMULA,
  'merror': ElementType.FORMULA,
  'mpadded': ElementType.FORMULA,
  'mphantom': ElementType.FORMULA,
  'mfenced': ElementType.FORMULA,
  'menclose': ElementType.FORMULA,
  'mtable': ElementType.FORMULA,
  'mtr': ElementType.FORMULA,
  'mtd': ElementType.FORMULA,
  'maligngroup': ElementType.FORMULA,
  'malignmark': ElementType.FORMULA,
  'mlabeledtr': ElementType.FORMULA,
  'maction': ElementType.FORMULA,
  'semantics': ElementType.FORMULA,
  'annotation': ElementType.FORMULA,
  'annotation-xml': ElementType.FORMULA,
};

// Comprehensive CSS class patterns that indicate element types
export const CSS_CLASS_PATTERNS: Array<{
  pattern: RegExp;
  elementType: ElementType;
}> = [
  // Navigation patterns
  { pattern: /\b(nav|menu|breadcrumb|sidebar|navigation|navbar|menubar)\b/i, elementType: ElementType.NAVIGATION },
  
  // Header/Footer patterns
  { pattern: /\b(header|masthead|banner|site-header|page-header|main-header)\b/i, elementType: ElementType.PAGE_HEADER },
  { pattern: /\b(footer|copyright|legal|site-footer|page-footer|main-footer)\b/i, elementType: ElementType.PAGE_FOOTER },
  { pattern: /\b(section-header|content-header)\b/i, elementType: ElementType.SECTION_HEADER },
  
  // Title and heading patterns
  { pattern: /\b(title|heading|headline|h[1-6]|header-text)\b/i, elementType: ElementType.TITLE },
  { pattern: /\b(subtitle|subheading|sub-title|sub-heading)\b/i, elementType: ElementType.SUB_HEADLINE },
  
  // Content patterns
  { pattern: /\b(content|article|post|story|narrative|text|body|main-content)\b/i, elementType: ElementType.NARRATIVE_TEXT },
  { pattern: /\b(paragraph|para|text-block)\b/i, elementType: ElementType.PARAGRAPH },
  { pattern: /\b(abstract|summary|synopsis)\b/i, elementType: ElementType.ABSTRACT },
  
  // List patterns
  { pattern: /\b(list|item|bullet|numbered|ordered|unordered)\b/i, elementType: ElementType.LIST_ITEM },
  
  // Form patterns - only match actual form containers, not styling divs
  { pattern: /\b(form-container|form-wrapper|contact-form|login-form|signup-form)\b/i, elementType: ElementType.FORM },
  { pattern: /\b(label|field-name|form-label)\b/i, elementType: ElementType.FIELD_NAME },
  { pattern: /\b(value|field-value|input-value)\b/i, elementType: ElementType.VALUE },
  
  // Table patterns
  { pattern: /\b(table|grid|data|tabular|spreadsheet)\b/i, elementType: ElementType.TABLE },
  
  // Media patterns
  { pattern: /\b(image|img|picture|photo|figure|media)\b/i, elementType: ElementType.IMAGE },
  { pattern: /\b(caption|img-caption|figure-caption|photo-caption)\b/i, elementType: ElementType.FIGURE_CAPTION },
  
  // Code patterns
  { pattern: /\b(code|highlight|syntax|language-|hljs|prettyprint|source|snippet)\b/i, elementType: ElementType.CODE_SNIPPET },
  
  // Address and email patterns removed - caused false positives
  
  // Mathematical patterns
  { pattern: /\b(math|latex|katex|mathjax|formula|equation)\b/i, elementType: ElementType.FORMULA },
  
  // Footnote patterns
  { pattern: /\b(footnote|endnote|note|reference)\b/i, elementType: ElementType.FOOTNOTE },
  
  // Page number patterns
  { pattern: /\b(page-number|pagination|page-info)\b/i, elementType: ElementType.PAGE_NUMBER },
  
  // Link patterns
  { pattern: /\b(link|hyperlink|url|href)\b/i, elementType: ElementType.LINK },
];

// Tags that should typically be ignored/filtered out
export const IGNORED_TAGS = new Set([
  'script',
  'style',
  'meta',
  'link',
  'noscript',
  'iframe', // we expand these automatically anyway before processing
  'object',
  'embed',
  'applet',
]);

// Tags that are typically navigation/UI elements
export const NAVIGATION_TAGS = new Set([
  'nav',
  'menu',
  'menuitem',
  'aside', // Often sidebars
]);

// Tags that typically contain metadata, not content
export const METADATA_TAGS = new Set([
  'head',
  'title', // Page title, not content title
  'meta',
  'link',
  'base',
]);

// Inline tags that should be preserved within text
export const INLINE_TAGS = new Set([
  'a',
  'strong',
  'b',
  'em',
  'i',
  'u',
  'span',
  'code',
  'kbd',
  'samp',
  'var',
  'mark',
  'small',
  'sub',
  'sup',
]);