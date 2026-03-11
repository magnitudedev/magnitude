/**
 * Advanced element classification logic
 * Comprehensive semantic analysis matching Unstructured Python library capabilities
 */

import * as cheerio from 'cheerio';
import { ElementType } from './types';
import { TAG_TO_ELEMENT_TYPE, CSS_CLASS_PATTERNS, INLINE_TAGS } from './mappings';

export class ElementClassifier {
  
  /**
   * Classify a DOM element based on its tag, attributes, and content
   */
  classifyElement($el: any): ElementType {
    const tagName = $el.prop('tagName')?.toLowerCase();
    if (!tagName) {
      return ElementType.UNCATEGORIZED_TEXT;
    }

    // Check direct tag mapping first for core structural elements
    const tagType = TAG_TO_ELEMENT_TYPE[tagName];
    if (tagType && ['table', 'form'].includes(tagName)) {
      return tagType;
    }

    // Then check for specialized elements
    const specializedType = this.classifySpecializedElement($el);
    if (specializedType !== ElementType.UNCATEGORIZED_TEXT) {
      return specializedType;
    }

    // Check other direct tag mappings
    if (tagType && tagType !== ElementType.TEXT) {
      return tagType;
    }

    // Check CSS classes for semantic hints
    const classType = this.classifyByCSS($el);
    if (classType !== ElementType.UNCATEGORIZED_TEXT) {
      return classType;
    }

    // Content-based classification for generic tags
    return this.classifyByContent($el);
  }

  /**
   * Classify specialized elements (forms, addresses, emails, code, etc.)
   */
  private classifySpecializedElement($el: any): ElementType {

    // Form elements
    if (this.isFormElement($el)) {
      return this.classifyFormElement($el);
    }

    // Code elements
    if (this.isCodeElement($el)) {
      return ElementType.CODE_SNIPPET;
    }

    // Simplified: treat specialized content as regular text for cleaner markdown
    // Address, email, formula, caption, footnote detection removed
    // This reduces false positives and focuses on what matters for markdown conversion

    // Page number elements
    if (this.isPageNumberElement($el)) {
      return ElementType.PAGE_NUMBER;
    }

    // Abstract elements
    if (this.isAbstractElement($el)) {
      return ElementType.ABSTRACT;
    }

    // Header/footer classification
    const headerFooterType = this.classifyHeaderFooter($el);
    if (headerFooterType !== ElementType.UNCATEGORIZED_TEXT) {
      return headerFooterType;
    }

    return ElementType.UNCATEGORIZED_TEXT;
  }

  /**
   * Check if element is a form-related element
   */
  private isFormElement($el: any): boolean {
    const tagName = $el.prop('tagName')?.toLowerCase();
    const formTags = ['form', 'input', 'textarea', 'select', 'button', 'fieldset', 'legend', 'label'];
    const type = $el.attr('type')?.toLowerCase();
    
    return formTags.includes(tagName || '') || 
           (tagName === 'input' && ['checkbox', 'radio', 'submit', 'button'].includes(type || ''));
  }

  /**
   * Classify form elements into specific types
   */
  private classifyFormElement($el: any): ElementType {
    const tagName = $el.prop('tagName')?.toLowerCase();
    const type = $el.attr('type')?.toLowerCase();
    const checked = $el.prop('checked');

    switch (tagName) {
      case 'form':
        return ElementType.FORM;
      case 'input':
        switch (type) {
          case 'checkbox':
            return checked ? ElementType.CHECK_BOX_CHECKED : ElementType.CHECK_BOX_UNCHECKED;
          case 'radio':
            return checked ? ElementType.RADIO_BUTTON_CHECKED : ElementType.RADIO_BUTTON_UNCHECKED;
          default:
            return ElementType.VALUE;
        }
      case 'label':
        return ElementType.FIELD_NAME;
      case 'textarea':
      case 'select':
        return ElementType.VALUE;
      case 'fieldset':
        // Fieldset should never be treated as a form - it's a form component
        return ElementType.TEXT;
      case 'legend':
        return ElementType.FIELD_NAME;
      case 'button':
        return ElementType.VALUE;
      default:
        // Only treat as form if it's actually a form element and not inside another form
        if (tagName === 'form') {
          return ElementType.FORM;
        }
        return ElementType.TEXT;
    }
  }

  /**
   * Check if element contains code
   */
  private isCodeElement($el: any): boolean {
    const tagName = $el.prop('tagName')?.toLowerCase();
    const className = $el.attr('class') || '';
    const codeTags = ['code', 'pre', 'kbd', 'samp', 'var'];
    const codeClasses = /\b(code|highlight|syntax|language-|hljs|prettyprint)\b/i;
    
    return codeTags.includes(tagName || '') || codeClasses.test(className);
  }

  /**
   * Check if element contains an address
   */
  private isAddressElement($el: any): boolean {
    const tagName = $el.prop('tagName')?.toLowerCase();
    const text = $el.text().trim();
    const className = $el.attr('class') || '';
    
    // Address tag
    if (tagName === 'address') {
      return true;
    }
    
    // Address-like classes (but not form-related)
    const addressClasses = /\b(address|location|postal)\b/i;
    const formClasses = /\b(form|contact-form|login-form|signup-form)\b/i;
    if (addressClasses.test(className) && !formClasses.test(className)) {
      return true;
    }
    
    // Specific contact class that's not form-related
    if (/\bcontact\b/i.test(className) && !/form/i.test(className)) {
      return true;
    }
    
    // Address patterns in text
    const addressPatterns = [
      /\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl)/i,
      /\b\d{5}(?:-\d{4})?\b/, // ZIP codes
      /\b[A-Z]{2}\s+\d{5}\b/, // State + ZIP
    ];
    
    return addressPatterns.some(pattern => pattern.test(text));
  }

  /**
   * Check if element contains an email address
   */
  private isEmailAddressElement($el: any): boolean {
    const text = $el.text().trim();
    
    // Email pattern
    const emailPattern = /\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b/;
    
    // Only classify as email if the text contains an email address
    // mailto: links with generic text should be classified as links
    return emailPattern.test(text);
  }

  /**
   * Check if element contains a mathematical formula
   */
  private isFormulaElement($el: any): boolean {
    const tagName = $el.prop('tagName')?.toLowerCase();
    const className = $el.attr('class') || '';
    const text = $el.text().trim();
    
    // MathML elements
    const mathTags = ['math', 'mrow', 'mi', 'mn', 'mo', 'mfrac', 'msup', 'msub'];
    if (mathTags.includes(tagName || '')) {
      return true;
    }
    
    // LaTeX-like classes
    const mathClasses = /\b(math|latex|katex|mathjax|formula|equation)\b/i;
    if (mathClasses.test(className)) {
      return true;
    }
    
    // Mathematical symbols and patterns
    const mathPatterns = [
      /[∑∏∫∂∇∆√∞±≤≥≠≈∈∉⊂⊃∪∩]/,
      /\$.*\$/,  // LaTeX delimiters
      /\\[a-zA-Z]+\{.*\}/, // LaTeX commands
    ];
    
    return mathPatterns.some(pattern => pattern.test(text));
  }

  /**
   * Check if element is a caption
   */
  private isCaptionElement($el: any): boolean {
    const tagName = $el.prop('tagName')?.toLowerCase();
    const className = $el.attr('class') || '';
    
    const captionTags = ['caption', 'figcaption'];
    const captionClasses = /\b(caption|figure-caption|img-caption)\b/i;
    
    return captionTags.includes(tagName || '') || captionClasses.test(className);
  }

  /**
   * Classify caption elements
   */
  private classifyCaptionElement($el: any): ElementType {
    const tagName = $el.prop('tagName')?.toLowerCase();
    
    if (tagName === 'figcaption') {
      return ElementType.FIGURE_CAPTION;
    }
    
    // Check if it's associated with a figure or image
    const parent = $el.parent();
    const parentTag = parent.prop('tagName')?.toLowerCase();
    
    if (parentTag === 'figure' || parent.find('img').length > 0) {
      return ElementType.FIGURE_CAPTION;
    }
    
    return ElementType.CAPTION;
  }

  /**
   * Check if element is a footnote
   */
  private isFootnoteElement($el: any): boolean {
    const className = $el.attr('class') || '';
    const id = $el.attr('id') || '';
    const text = $el.text().trim();
    
    const footnoteClasses = /\b(footnote|endnote|note)\b/i;
    const footnoteIds = /\b(fn|footnote|note)-?\d+\b/i;
    const footnotePatterns = /^\[\d+\]|\(\d+\)|^\d+\./;
    
    return footnoteClasses.test(className) || 
           footnoteIds.test(id) || 
           footnotePatterns.test(text);
  }

  /**
   * Check if element contains a page number
   */
  private isPageNumberElement($el: any): boolean {
    const text = $el.text().trim();
    const className = $el.attr('class') || '';
    
    const pageClasses = /\b(page-?number|pagination)\b/i;
    const pagePatterns = [
      /^Page\s+\d+$/i,
      /^\d+\s+of\s+\d+$/i,
      /^\d+\s*\/\s*\d+$/,
      /^-?\s*\d+\s*-?$/,
    ];
    
    return pageClasses.test(className) || 
           pagePatterns.some(pattern => pattern.test(text));
  }

  /**
   * Check if element is an abstract
   */
  private isAbstractElement($el: any): boolean {
    const className = $el.attr('class') || '';
    const id = $el.attr('id') || '';
    const text = $el.text().trim();
    
    const abstractClasses = /\b(abstract|summary)\b/i;
    const abstractIds = /\babstract\b/i;
    const abstractStart = /^abstract\b/i;
    
    return abstractClasses.test(className) || 
           abstractIds.test(id) || 
           abstractStart.test(text);
  }

  /**
   * Classify header and footer elements
   */
  private classifyHeaderFooter($el: any): ElementType {
    const tagName = $el.prop('tagName')?.toLowerCase();
    const className = $el.attr('class') || '';
    const id = $el.attr('id') || '';
    
    // Direct tag classification
    if (tagName === 'header') {
      return this.isPageLevelHeader($el) ? ElementType.PAGE_HEADER : ElementType.HEADER;
    }
    
    if (tagName === 'footer') {
      return this.isPageLevelFooter($el) ? ElementType.PAGE_FOOTER : ElementType.FOOTER;
    }
    
    // Class-based classification
    const headerClasses = /\b(header|masthead|banner)\b/i;
    const footerClasses = /\b(footer|contentinfo)\b/i;
    const pageHeaderClasses = /\b(page-header|site-header|main-header)\b/i;
    const pageFooterClasses = /\b(page-footer|site-footer|main-footer)\b/i;
    
    const combinedClasses = `${className} ${id}`;
    
    if (pageHeaderClasses.test(combinedClasses)) {
      return ElementType.PAGE_HEADER;
    }
    
    if (pageFooterClasses.test(combinedClasses)) {
      return ElementType.PAGE_FOOTER;
    }
    
    if (headerClasses.test(combinedClasses)) {
      return ElementType.HEADER;
    }
    
    if (footerClasses.test(combinedClasses)) {
      return ElementType.FOOTER;
    }
    
    return ElementType.UNCATEGORIZED_TEXT;
  }

  /**
   * Check if header is page-level
   */
  private isPageLevelHeader($el: any): boolean {
    // Check if it's a direct child of body or main container
    const parent = $el.parent();
    const parentTag = parent.prop('tagName')?.toLowerCase();
    
    return ['body', 'main', 'html'].includes(parentTag || '') ||
           parent.hasClass('page') ||
           parent.hasClass('container') ||
           parent.hasClass('wrapper');
  }

  /**
   * Check if footer is page-level
   */
  private isPageLevelFooter($el: any): boolean {
    // Similar logic to page-level header
    const parent = $el.parent();
    const parentTag = parent.prop('tagName')?.toLowerCase();
    
    return ['body', 'main', 'html'].includes(parentTag || '') ||
           parent.hasClass('page') ||
           parent.hasClass('container') ||
           parent.hasClass('wrapper');
  }

  /**
   * Classify element based on CSS classes
   */
  private classifyByCSS($el: any): ElementType {
    const className = $el.attr('class') || '';
    const id = $el.attr('id') || '';
    const combinedClasses = `${className} ${id}`.toLowerCase();

    for (const { pattern, elementType } of CSS_CLASS_PATTERNS) {
      if (pattern.test(combinedClasses)) {
        return elementType;
      }
    }

    return ElementType.UNCATEGORIZED_TEXT;
  }

  /**
   * Classify element based on content analysis
   */
  private classifyByContent($el: any): ElementType {
    const text = $el.text().trim();
    const tagName = $el.prop('tagName')?.toLowerCase();

    // Empty or very short content
    if (text.length < 3) {
      return ElementType.UNCATEGORIZED_TEXT;
    }

    // Check if it looks like a title/heading
    if (this.looksLikeTitle(text, $el)) {
      return ElementType.TITLE;
    }

    // Check if it looks like a list item
    if (this.looksLikeListItem(text, $el)) {
      return ElementType.LIST_ITEM;
    }

    // Default classification based on tag
    switch (tagName) {
      case 'div':
      case 'section':
      case 'article':
        // Long content in block elements is likely narrative text
        return text.length > 50 ? ElementType.NARRATIVE_TEXT : ElementType.TEXT;
      
      case 'span':
        // Spans are usually inline text
        return ElementType.TEXT;
      
      case 'p':
        return ElementType.NARRATIVE_TEXT;
      
      default:
        return ElementType.TEXT;
    }
  }

  /**
   * Heuristics to determine if text looks like a title/heading
   */
  private looksLikeTitle(text: string, $el: any): boolean {
    // Short text (likely titles are usually under 100 chars)
    if (text.length > 100) {
      return false;
    }

    // Check for title-like formatting
    const hasCapitalization = /^[A-Z]/.test(text) && text === text.charAt(0).toUpperCase() + text.slice(1);
    const hasColonOrDash = /[:\\-–—]/.test(text);
    const isAllCaps = text === text.toUpperCase() && text.length > 3;
    const hasNumbers = /^\\d+\\.?\\s/.test(text); // "1. Title" or "1 Title"

    // Check parent context
    const parent = $el.parent();
    const parentTag = parent.prop('tagName')?.toLowerCase();
    const isInHeader = parentTag === 'header' || parent.closest('header').length > 0;

    // Check styling hints
    const style = $el.attr('style') || '';
    const hasBoldStyle = /font-weight\\s*:\\s*(bold|[6-9]00)/i.test(style);
    const hasLargeFont = /font-size\\s*:\\s*([2-9]\\d|\\d{3,})px/i.test(style);

    return (
      hasCapitalization ||
      hasColonOrDash ||
      isAllCaps ||
      hasNumbers ||
      isInHeader ||
      hasBoldStyle ||
      hasLargeFont
    );
  }

  /**
   * Heuristics to determine if text looks like a list item
   */
  private looksLikeListItem(text: string, $el: any): boolean {
    // Check for bullet-like prefixes
    const bulletPatterns = [
      /^[•·▪▫‣⁃]\\s/, // Unicode bullets
      /^[-*+]\\s/, // ASCII bullets
      /^\\d+\\.\\s/, // Numbered lists
      /^[a-zA-Z]\\.\\s/, // Lettered lists
      /^\\([a-zA-Z0-9]+\\)\\s/, // Parenthetical lists
    ];

    const startsWithBullet = bulletPatterns.some(pattern => pattern.test(text));

    // Check parent context
    const parent = $el.parent();
    const parentTag = parent.prop('tagName')?.toLowerCase();
    const isInList = ['ul', 'ol', 'dl'].includes(parentTag || '');

    // Check siblings - if most siblings also look like list items
    const siblings = parent.children().not($el);
    const siblingTexts = siblings.map((_: any, el: any) => cheerio.load(el).text().trim()).get();
    const siblingsWithBullets = siblingTexts.filter((siblingText: string) => 
      bulletPatterns.some(pattern => pattern.test(siblingText))
    ).length;
    const mostSiblingsAreBullets = siblingsWithBullets > siblings.length * 0.5;

    return startsWithBullet || isInList || mostSiblingsAreBullets;
  }

  /**
   * Check if element should be treated as inline (part of parent's text)
   */
  isInlineElement($el: any): boolean {
    const tagName = $el.prop('tagName')?.toLowerCase();
    return INLINE_TAGS.has(tagName || '');
  }

  /**
   * Extract clean text from element, handling inline elements appropriately
   */
  extractCleanText($el: any): string {
    // Clone the element to avoid modifying original
    const $clone = $el.clone();

    // Replace inline elements with their text content
    $clone.find('*').each((_: any, child: any) => {
      const $child = cheerio.load(child);
      const childEl = $child.root().children().first();
      
      if (this.isInlineElement(childEl)) {
        // Keep the text but remove the tag
        childEl.replaceWith(childEl.text());
      }
    });

    // Get text and normalize whitespace
    return $clone.text()
      .replace(/\\s+/g, ' ')
      .trim();
  }
}