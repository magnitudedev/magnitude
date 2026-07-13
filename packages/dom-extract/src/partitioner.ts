/**
 * Advanced DOM partitioner with comprehensive element extraction
 * Matching Unstructured Python library capabilities
 */

import * as cheerio from 'cheerio';
import { createId } from '@magnitudedev/generate-id';
import type { AnyNode } from 'domhandler';


import { 
  ElementType,
  ProcessingStrategy,
  ChunkingStrategy
} from './types';
import type { 
  Element as UnstructuredElement, 
  TableElement, 
  ImageElement,
  FormElement,
  CheckBoxElement,
  RadioButtonElement,
  LinkElement,
  CodeElement,
  FormulaElement,
  CompositeElement,
  ElementMetadata,
  PartitionOptions, 
  PartitionResult,
  AnyElement
} from './types';
import { DOMCleaner } from './cleaner';
import { ElementClassifier } from './classifier';
import { ContentHandlers } from './content-handlers';

export class DOMPartitioner {
  private cleaner: DOMCleaner;
  private classifier: ElementClassifier;
  private contentHandlers: ContentHandlers;
  private options: PartitionOptions;
  private createId = createId;

  constructor(options: PartitionOptions = {}) {
    this.options = {
      skipNavigation: true,
      skipHeaders: false,
      skipFooters: false,
      skipForms: false,
      skipHeadersAndFooters: false,
      minTextLength: 3,
      maxTextLength: undefined,
      preserveWhitespace: false,
      extractTables: true,
      inferTableStructure: true,
      skipInferTableTypes: [],
      extractImages: true,
      includeImageAlt: true,
      extractImageBlockTypes: [],
      extractImageBlockToPayload: false,
      extractImageBlockOutputDir: undefined,
      extractForms: true,
      extractFormFields: true,
      extractLinks: true,
      languages: undefined,
      detectLanguagePerElement: false,
      includeCoordinates: false,
      coordinateSystem: undefined,
      includePageBreaks: true,
      maintainHierarchy: true,
      strategy: ProcessingStrategy.AUTO,
      chunkingStrategy: ChunkingStrategy.NONE,
      maxCharacters: undefined,
      newAfterNChars: undefined,
      combineTextUnderNChars: undefined,
      includeOriginalHtml: false,
      includeMetadata: true,
      metadataFilename: undefined,
      uniqueElementIds: false,
      processAttachments: false,
      attachmentPartitioningStrategy: ProcessingStrategy.AUTO,
      elementTypeFilters: undefined,
      contentFilters: undefined,
      includeDebugMetadata: false,
      detectionOrigin: undefined,
      ...options,
    };
    
    this.cleaner = new DOMCleaner(this.options);
    this.classifier = new ElementClassifier();
    this.contentHandlers = new ContentHandlers();
  }

  /**
   * Partition HTML content into structured elements
   */
  partition(html: string): PartitionResult {
    const startTime = performance.now();
    const warnings: string[] = [];

    try {
      // Load HTML with cheerio
      const $ = cheerio.load(html, {
        xmlMode: false,
      });

      // Clean the DOM
      this.cleaner.clean($);

      // Extract elements
      const elements = this.extractElements($);

      const processingTime = Math.max(1, Math.round(performance.now() - startTime));

      // Calculate comprehensive metadata
      const elementTypeCounts: Record<ElementType, number> = {} as Record<ElementType, number>;
      let totalTextLength = 0;
      let tablesExtracted = 0;
      let imagesExtracted = 0;
      let formsExtracted = 0;
      let linksExtracted = 0;

      elements.forEach(element => {
        elementTypeCounts[element.type] = (elementTypeCounts[element.type] || 0) + 1;
        totalTextLength += element.text.length;
        
        if (element.type === ElementType.TABLE) tablesExtracted++;
        if ([ElementType.IMAGE, ElementType.PICTURE, ElementType.FIGURE].includes(element.type)) imagesExtracted++;
        if (element.type === ElementType.FORM) formsExtracted++;
        if (element.type === ElementType.LINK) linksExtracted++;
      });

      return {
        elements,
        metadata: {
          totalElements: elements.length,
          processingTime,
          warnings: warnings.length > 0 ? warnings : undefined,
          errors: undefined,
          elementTypeCounts,
          averageElementLength: elements.length > 0 ? Math.round(totalTextLength / elements.length) : 0,
          tablesExtracted,
          imagesExtracted,
          formsExtracted,
          linksExtracted,
          detectedLanguages: this.options.languages,
        },
      };
    } catch (error) {
      warnings.push(`Partitioning error: ${error instanceof Error ? error.message : 'Unknown error'}`);
      
      return {
        elements: [],
        metadata: {
          totalElements: 0,
          processingTime: Math.max(1, Math.round(performance.now() - startTime)),
          warnings,
        },
      };
    }
  }

  /**
   * Extract structured elements from cleaned DOM
   */
  private extractElements($: cheerio.CheerioAPI): UnstructuredElement[] {
    const elements: UnstructuredElement[] = [];
    const processedElements = new Set<any>();

    // Find all meaningful elements in document order
    // Use body if it exists, otherwise search all elements
    if ($('body').length > 0) {
      $('body').find('*').each((_, element) => {
        this.processElement($, element, processedElements, elements);
      });
    } else {
      $('*').each((_, element) => {
        this.processElement($, element, processedElements, elements);
      });
    }

    return elements;
  }

  private processElement($: cheerio.CheerioAPI, element: AnyNode, processedElements: Set<AnyNode>, elements: UnstructuredElement[]): void {
    if (processedElements.has(element)) {
      return;
    }

    const $el = $(element);
    const elementType = this.classifier.classifyElement($el);

    // Skip uncategorized elements with no meaningful content
    if (elementType === ElementType.UNCATEGORIZED_TEXT) {
      processedElements.add(element);
      return;
    }



    // Handle different element types
    let extractedElement: UnstructuredElement | null = null;

    switch (elementType) {
      case ElementType.TABLE:
        extractedElement = this.extractTable($, $el);
        // Only mark table children as processed if we actually extracted a table
        // For layout tables (where extractTable returns null), let children be processed individually
        if (extractedElement) {
          $el.find('*').each((_: any, child: any) => {
            processedElements.add(child);
          });
        }
        break;
        
      case ElementType.IMAGE:
      case ElementType.PICTURE:
      case ElementType.FIGURE:
        extractedElement = this.extractImage($, $el);
        break;
        
      case ElementType.FORM:
        if (this.options.extractForms) {
          extractedElement = this.extractForm($, $el);
          // Don't mark children as processed - let them be processed individually too
          // This allows both form-level and field-level extraction
        }
        break;
        
      case ElementType.CHECK_BOX_CHECKED:
      case ElementType.CHECK_BOX_UNCHECKED:
        extractedElement = this.extractCheckBox($, $el);
        break;
        
      case ElementType.RADIO_BUTTON_CHECKED:
      case ElementType.RADIO_BUTTON_UNCHECKED:
        extractedElement = this.extractRadioButton($, $el);
        break;
        
      case ElementType.VALUE:
        extractedElement = this.extractValue($, $el);
        break;
        
      case ElementType.LINK:
        if (this.options.extractLinks) {
          extractedElement = this.extractLink($, $el);
        }
        break;
        
      // ADDRESS and EMAIL_ADDRESS cases removed - caused false positives
        
      case ElementType.CODE_SNIPPET:
        extractedElement = this.extractCode($, $el);
        break;
        
      case ElementType.FORMULA:
        extractedElement = this.extractFormula($, $el);
        break;
        
      case ElementType.PAGE_BREAK:
        if (this.options.includePageBreaks) {
          extractedElement = this.extractPageBreak($, $el);
        }
        break;

      default:
        extractedElement = this.extractTextElement($, $el, elementType);
        break;
    }

    processedElements.add(element);
    
    if (extractedElement) {
      elements.push(extractedElement);
    }
  }

  /**
   * Extract a text-based element
   */
  private extractTextElement($: cheerio.CheerioAPI, $el: cheerio.Cheerio<AnyNode>, elementType: ElementType): UnstructuredElement | null {
    const text = this.classifier.extractCleanText($el);
    
    if (text.length < this.options.minTextLength!) {
      return null;
    }

    // Check max text length if specified
    if (this.options.maxTextLength && text.length > this.options.maxTextLength) {
      return null;
    }

    const metadata = this.extractMetadata($, $el);

    return {
      id: this.createId(),
      type: elementType,
      text,
      metadata: metadata,
    };
  }

  /**
   * Extract table element with structure
   */
  private extractTable($: cheerio.CheerioAPI, $el: cheerio.Cheerio<AnyNode>): TableElement | null {
    if (!this.options.extractTables) {
      return null;
    }

    // Detect layout tables vs data tables
    if (this.isLayoutTable($, $el)) {
      // For layout tables, process their content but don't create a table element
      // This allows extraction of content within layout tables
      return null;
    }

    let rows: string[][] = [];
    let headers: string[] | undefined;

    // Extract headers from thead or first row with th elements
    const $thead = $el.find('thead tr').first();
    if ($thead.length > 0) {
      headers = $thead.find('th, td').map((_: any, cell: any) => {
        return $(cell).text().trim();
      }).get();
    } else {
      // Check if first row has th elements (header cells)
      const $firstRow = $el.find('tr').first();
      if ($firstRow.length > 0 && $firstRow.find('th').length > 0) {
        headers = $firstRow.find('th, td').map((_: any, cell: any) => {
          return $(cell).text().trim();
        }).get();
      }
    }

    // Extract data rows
    const $rows = $el.find('tbody tr, tr').filter((_: any, row: any) => {
      // Skip header row if we already extracted headers from thead or th elements
      const isInThead = $(row).closest('thead').length > 0;
      const hasThElements = $(row).find('th').length > 0;
      return !headers || (!isInThead && !hasThElements);
    });

    $rows.each((_: any, row: any) => {
      const $row = $(row);
      const cells = $row.find('td, th').map((_: any, cell: any) => {
        return $(cell).text().trim();
      }).get();
      
      if (cells.length > 0) {
        rows.push(cells);
      }
    });

    // Allow empty tables - they should still be extracted as table elements

    // If no explicit headers but first row looks like headers
    if (!headers && rows.length > 1) {
      const firstRow = rows[0];
      const secondRow = rows[1];
      
      // Heuristic: treat first row as headers if it contains text and second row contains numbers
      const firstRowHasAlpha = firstRow.some(cell => /[a-zA-Z]/.test(cell));
      const secondRowHasNumbers = secondRow.some(cell => /\d/.test(cell));
      
      // If first row is text-based and second row has numbers, treat first as headers
      if (firstRowHasAlpha && secondRowHasNumbers) {
        headers = firstRow;
        rows.shift(); // Remove first row from data
      }
    }

    // Normalize rows to match header count if headers exist
    if (headers && headers.length > 0) {
      const normalizedRows = rows.map(row => {
        const normalizedRow = [...row];
        // Pad with empty strings if row is shorter than headers
        while (normalizedRow.length < headers.length) {
          normalizedRow.push('');
        }
        // Truncate if row is longer than headers
        return normalizedRow.slice(0, headers.length);
      });
      rows = normalizedRows;
    }

    const text = this.generateTableText(rows, headers);
    const metadata = this.extractMetadata($, $el);

    return {
      id: this.createId(),
      type: ElementType.TABLE,
      text,
      metadata: metadata,
      rows,
      headers: headers || [],
    };
  }

  /**
   * Detect if a table is used for layout rather than data
   * 
   * Note: False positives (data tables classified as layout) lose tabular structure
   * but preserve all content as individual elements. False negatives (layout tables
   * treated as data) cause massive duplication and unusable output.
   */
  private isLayoutTable($: cheerio.CheerioAPI, $el: cheerio.Cheerio<AnyNode>): boolean {
    // Check for common layout table indicators
    const hasHeaders = $el.find('th').length > 0 || $el.find('thead').length > 0;
    const hasComplexLayout = $el.html()?.includes('colspan') || $el.html()?.includes('rowspan');
    const hasLayoutAttributes = $el.attr('cellpadding') || $el.attr('cellspacing') || $el.attr('border');
    
    // Count rows and check for consistent structure
    const rows = $el.find('tr');
    if (rows.length === 0) return true; // Empty table is likely layout
    
    // Check column consistency - data tables usually have consistent columns
    const cellCounts: number[] = [];
    rows.each((_: any, row: any) => {
      const cellCount = $(row).find('td, th').length;
      cellCounts.push(cellCount);
    });
    
    const uniqueCellCounts = [...new Set(cellCounts)];
    const hasInconsistentColumns = uniqueCellCounts.length > 3; // More lenient - allow more variation
    
    // Check if table contains mostly non-tabular content (links, images, forms)
    const totalCells = $el.find('td, th').length;
    const cellsWithLinks = $el.find('td a, th a').length;
    const cellsWithImages = $el.find('td img, th img').length;
    const cellsWithForms = $el.find('td form, th form, td input, th input').length;
    const nonTabularContent = cellsWithLinks + cellsWithImages + cellsWithForms;
    const hasHighNonTabularRatio = totalCells > 0 && (nonTabularContent / totalCells) > 0.3;
    
    // Strong layout indicators (high confidence)
    if (hasLayoutAttributes && !hasHeaders && hasInconsistentColumns) {
      return true;
    }
    
    // Medium confidence: complex layout without headers and high non-tabular content
    if (!hasHeaders && hasComplexLayout && hasHighNonTabularRatio) {
      return true;
    }
    
    // Conservative: only flag as layout if multiple strong indicators
    if (!hasHeaders && hasInconsistentColumns && hasHighNonTabularRatio && rows.length > 20) {
      return true;
    }
    
    return false;
  }

  /**
   * Extract image element
   */
  private extractImage($: cheerio.CheerioAPI, $el: cheerio.Cheerio<AnyNode>): ImageElement | null {
    if (!this.options.extractImages) {
      return null;
    }

    const src = $el.attr('src');
    const alt = $el.attr('alt') || '';
    const width = parseInt($el.attr('width') || '0') || undefined;
    const height = parseInt($el.attr('height') || '0') || undefined;

    const text = this.options.includeImageAlt ? alt : '';
    const metadata = this.extractMetadata($, $el);

    return {
      id: this.createId(),
      type: ElementType.IMAGE,
      text,
      metadata: metadata,
      src,
      alt,
      width,
      height,
    };
  }

  /**
   * Extract metadata from DOM element
   */
  private extractMetadata($: cheerio.CheerioAPI, $el: cheerio.Cheerio<AnyNode>): ElementMetadata {
    const tagName = $el.prop('tagName')?.toLowerCase();
    const classAttr = $el.attr('class');
    const cssClasses = classAttr ? classAttr.split(/\s+/).filter(Boolean) : undefined;
    const elementId = $el.attr('id');
    const text = $el.text();

    const metadata: ElementMetadata = {
      tagName,
      cssClasses: cssClasses && cssClasses.length > 0 ? cssClasses : undefined,
      elementId: elementId && elementId.trim() !== '' ? elementId : undefined,
      textLength: text.length,
    };

    // Extract links if enabled
    if (this.options.extractLinks) {
      const links = this.contentHandlers.extractLinks($, $el);
      if (links.length > 0) {
        metadata.links = links;
        metadata.linkTexts = links.map(link => link.text);
        metadata.linkUrls = links.map(link => link.url);
      }
    }

    // Extract emphasis information
    const emphasis = this.contentHandlers.extractEmphasis($, $el);
    if (emphasis.contents.length > 0) {
      metadata.emphasizedTextContents = emphasis.contents;
      metadata.emphasizedTextTags = emphasis.tags;
    }

    // Extract coordinates if enabled
    if (this.options.includeCoordinates) {
      const coordinates = this.contentHandlers.extractCoordinates($el);
      if (coordinates) {
        metadata.coordinates = coordinates;
      }
    }

    // Include original HTML if requested
    if (this.options.includeOriginalHtml) {
      metadata.originalHtml = $.html($el) || undefined;
    }

    return metadata;
  }

  /**
   * Extract form element with fields
   */
  private extractForm($: cheerio.CheerioAPI, $el: cheerio.Cheerio<AnyNode>): FormElement | null {
    const fields = this.options.extractFormFields ? this.contentHandlers.extractFormFields($, $el) : [];
    const text = fields.length > 0 
      ? fields.map(f => `${f.fieldName}: ${f.fieldValue}`).join('; ')
      : $el.text().trim() || 'Form';
    const metadata = this.extractMetadata($, $el);

    return {
      id: this.createId(),
      type: ElementType.FORM,
      text,
      metadata: metadata,
      fields: this.options.extractFormFields ? fields : undefined
    };
  }

  /**
   * Extract checkbox element
   */
  private extractCheckBox($: cheerio.CheerioAPI, $el: cheerio.Cheerio<AnyNode>): CheckBoxElement | null {
    const checked = Boolean($el.prop('checked'));
    const value = $el.attr('value') || '';
    const label = $el.prev('label').text().trim() || $el.next('label').text().trim() || '';
    const text = label || (checked ? 'checked' : 'unchecked');
    const metadata = this.extractMetadata($, $el);

    return {
      id: this.createId(),
      type: checked ? ElementType.CHECK_BOX_CHECKED : ElementType.CHECK_BOX_UNCHECKED,
      text,
      metadata: metadata,
      checked,
      value: value || undefined
    };
  }

  /**
   * Extract radio button element
   */
  private extractRadioButton($: cheerio.CheerioAPI, $el: cheerio.Cheerio<AnyNode>): RadioButtonElement | null {
    const checked = Boolean($el.prop('checked'));
    const value = $el.attr('value') || '';
    const groupName = $el.attr('name') || '';
    const label = $el.prev('label').text().trim() || $el.next('label').text().trim() || '';
    const text = label || (checked ? 'selected' : 'unselected');
    const metadata = this.extractMetadata($, $el);

    return {
      id: this.createId(),
      type: checked ? ElementType.RADIO_BUTTON_CHECKED : ElementType.RADIO_BUTTON_UNCHECKED,
      text,
      metadata: metadata,
      checked,
      value: value || undefined,
      groupName: groupName || undefined
    };
  }

  /**
   * Extract value element (form inputs, buttons, etc.)
   */
  private extractValue($: cheerio.CheerioAPI, $el: cheerio.Cheerio<AnyNode>): UnstructuredElement | null {
    const tagName = $el.prop('tagName')?.toLowerCase();
    const type = $el.attr('type')?.toLowerCase();
    let text = '';
    
    // Extract appropriate text based on element type
    switch (tagName) {
      case 'input':
        const value = $el.attr('value') || $el.val() || '';
        text = value.toString();
        break;
      case 'button':
        text = $el.text().trim() || $el.attr('value') || '';
        break;
      case 'select':
        const selectedOption = $el.find('option:selected');
        text = selectedOption.text().trim() || selectedOption.attr('value') || '';
        break;
      case 'textarea':
        text = $el.text().trim() || $el.val()?.toString() || '';
        break;
      case 'option':
        text = $el.text().trim() || $el.attr('value') || '';
        break;
      default:
        text = $el.text().trim() || $el.attr('value') || '';
    }
    
    // Don't filter by text length for form elements - they might have empty values
    const metadata = this.extractMetadata($, $el);

    return {
      id: this.createId(),
      type: ElementType.VALUE,
      text,
      metadata: metadata,
    };
  }

  /**
   * Extract link element
   */
  private extractLink($: cheerio.CheerioAPI, $el: cheerio.Cheerio<AnyNode>): LinkElement | null {
    const url = $el.attr('href') || '';
    const linkText = $el.text().trim();
    
    if (!url || !linkText) {
      return null;
    }

    const metadata = this.extractMetadata($, $el);

    return {
      id: this.createId(),
      type: ElementType.LINK,
      text: linkText,
      metadata: metadata,
      url,
      linkText
    };
  }

  // extractAddress and extractEmailAddress methods removed - caused false positives

  /**
   * Extract code element
   */
  private extractCode($: cheerio.CheerioAPI, $el: cheerio.Cheerio<AnyNode>): CodeElement | null {
    const codeBlock = $el.text();
    const language = this.contentHandlers.detectCodeLanguage($el);
    const metadata = this.extractMetadata($, $el);

    return {
      id: this.createId(),
      type: ElementType.CODE_SNIPPET,
      text: codeBlock,
      metadata: metadata,
      language,
      codeBlock
    };
  }

  /**
   * Extract formula element
   */
  private extractFormula($: cheerio.CheerioAPI, $el: cheerio.Cheerio<AnyNode>): FormulaElement | null {
    const formula = $el.text();
    const formulaType = this.contentHandlers.detectFormulaType($el);
    const metadata = this.extractMetadata($, $el);

    return {
      id: this.createId(),
      type: ElementType.FORMULA,
      text: formula,
      metadata: metadata,
      formula,
      formulaType
    };
  }

  /**
   * Extract page break element
   */
  private extractPageBreak($: cheerio.CheerioAPI, $el: cheerio.Cheerio<AnyNode>): UnstructuredElement | null {
    const metadata = this.extractMetadata($, $el);

    return {
      id: this.createId(),
      type: ElementType.PAGE_BREAK,
      text: '---',
      metadata
    };
  }

  /**
   * Generate readable text representation of table
   */
  private generateTableText(rows: string[][], headers?: string[]): string {
    const lines: string[] = [];

    if (headers) {
      lines.push(headers.join(' | '));
      lines.push(headers.map(() => '---').join(' | '));
    }

    rows.forEach(row => {
      lines.push(row.join(' | '));
    });

    return lines.join('\\n');
  }
}