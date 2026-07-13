/**
 * DOM cleaning utilities
 * Removes unwanted elements and normalizes HTML structure
 */

import * as cheerio from 'cheerio';
import { IGNORED_TAGS, NAVIGATION_TAGS } from './mappings';
import type { PartitionOptions } from './types';

export class DOMCleaner {
  private options: PartitionOptions;

  constructor(options: PartitionOptions = {}) {
    this.options = options;
  }

  /**
   * Clean and normalize HTML content
   */
  clean($: cheerio.CheerioAPI): cheerio.CheerioAPI {
    // Remove script, style, and other unwanted tags
    this.removeIgnoredTags($);
    
    // Remove navigation elements if requested
    if (this.options.skipNavigation) {
      this.removeNavigationElements($);
    }
    
    // Remove headers/footers if requested
    if (this.options.skipHeaders) {
      this.removeHeaders($);
    }
    if (this.options.skipFooters) {
      this.removeFooters($);
    }
    if (this.options.skipHeadersAndFooters) {
      this.removeHeaders($);
      this.removeFooters($);
    }
    
    // Remove forms if requested
    if (this.options.skipForms) {
      this.removeForms($);
    }
    
    // Clean up whitespace and empty elements
    this.normalizeWhitespace($);
    this.removeEmptyElements($);
    
    return $;
  }

  /**
   * Remove script, style, and other ignored tags
   */
  private removeIgnoredTags($: cheerio.CheerioAPI): void {
    IGNORED_TAGS.forEach(tag => {
      $(tag).remove();
    });
    
    // Remove comments
    $('*').contents().filter(function() {
      return this.type === 'comment';
    }).remove();
  }

  /**
   * Remove navigation elements
   */
  private removeNavigationElements($: cheerio.CheerioAPI): void {
    // Remove by tag
    NAVIGATION_TAGS.forEach(tag => {
      $(tag).remove();
    });
    
    // Remove by common navigation classes/IDs
    const navSelectors = [
      '[class*="nav"]',
      '[class*="menu"]',
      '[class*="breadcrumb"]',
      '[class*="sidebar"]',
      '[id*="nav"]',
      '[id*="menu"]',
      '[role="navigation"]',
      '[role="menu"]',
      '[role="menubar"]',
    ];
    
    navSelectors.forEach(selector => {
      $(selector).remove();
    });
  }

  /**
   * Remove header elements
   */
  private removeHeaders($: cheerio.CheerioAPI): void {
    const headerSelectors = [
      'header',
      '[class*="header"]',
      '[class*="masthead"]',
      '[class*="banner"]',
      '[id*="header"]',
      '[role="banner"]',
    ];
    
    headerSelectors.forEach(selector => {
      $(selector).remove();
    });
  }

  /**
   * Remove footer elements
   */
  private removeFooters($: cheerio.CheerioAPI): void {
    const footerSelectors = [
      'footer',
      '[class*="footer"]',
      '[class*="copyright"]',
      '[class*="legal"]',
      '[id*="footer"]',
      '[role="contentinfo"]',
    ];
    
    footerSelectors.forEach(selector => {
      $(selector).remove();
    });
  }

  /**
   * Remove form elements
   */
  private removeForms($: cheerio.CheerioAPI): void {
    $('form, input, button, select, textarea').remove();
  }

  /**
   * Normalize whitespace in text content
   */
  private normalizeWhitespace($: cheerio.CheerioAPI): void {
    if (this.options.preserveWhitespace) {
      return;
    }

    // Normalize whitespace in text nodes
    $('*').contents().filter(function() {
      return this.type === 'text';
    }).each((_, node) => {
      if ('data' in node && node.data) {
        // Replace multiple whitespace with single space
        node.data = node.data.replace(/\\s+/g, ' ');
      }
    });
  }

  /**
   * Remove empty elements that don't contribute content
   */
  private removeEmptyElements($: cheerio.CheerioAPI): void {
    // Remove elements with no text content and no meaningful children
    $('*').each((_, element) => {
      const $el = $(element);
      const text = $el.text().trim();
      const tagName = (element as any).tagName;
      const hasImages = $el.find('img').length > 0;
      const hasTables = $el.find('table').length > 0;
      const hasInputs = $el.find('input, button, select, textarea').length > 0;
      
      // Keep elements that have:
      // - Text content above minimum length
      // - Are heading elements (always keep)
      // - Are table cells (always keep)
      // - Are images (always keep if extracting)
      // - Images (if extracting images)
      // - Tables (if extracting tables)
      // - Form inputs (if not skipping forms)
      const isHeading = /^h[1-6]$/i.test(tagName);
      const isTableCell = /^(td|th)$/i.test(tagName);
      const isImage = tagName === 'img';
      const isFormElement = /^(input|button|select|textarea|form|label|fieldset|legend)$/i.test(tagName);
      const isTable = tagName === 'table';
      const shouldKeep = 
        text.length >= (this.options.minTextLength || 3) ||
        isHeading ||
        isTableCell ||
        (isImage && this.options.extractImages) ||
        (hasImages && this.options.extractImages) ||
        (hasTables && this.options.extractTables) ||
        (isTable && this.options.extractTables) ||
        (hasInputs && !this.options.skipForms) ||
        (isFormElement && !this.options.skipForms);

      
      if (!shouldKeep && !this.hasSignificantChildren($el)) {
        $el.remove();
      }
    });
  }

  /**
   * Check if element has children that should be preserved
   */
  private hasSignificantChildren($el: any): boolean {
    const significantTags = ['img', 'table', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'form', 'input', 'textarea', 'select', 'button', 'td', 'th', 'tr'];
    return significantTags.some(tag => $el.find(tag).length > 0);
  }
}