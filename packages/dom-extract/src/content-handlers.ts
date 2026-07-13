/**
 * Content-specific handlers for advanced processing
 * Handles specialized content types like forms, addresses, emails, etc.
 */

// Cheerio is passed as parameter to methods
import { 
  FormField, 
  Link, 
  Point, 
  CoordinatesMetadata
} from './types';

export class ContentHandlers {
  
  /**
   * Extract form fields and their values
   */
  extractFormFields($: any, $form: any): FormField[] {
    const fields: FormField[] = [];
    
    // Find all form inputs
    $form.find('input, textarea, select').each((_: any, element: any) => {
      const $el = $(element);
      const fieldType = $el.attr('type') || $el.prop('tagName')?.toLowerCase();
      const fieldName = $el.attr('name') || $el.attr('id') || $el.prev('label').text().trim() || 'field';
      let fieldValue = '';
      
      switch (fieldType) {
        case 'checkbox':
        case 'radio':
          fieldValue = $el.prop('checked') ? ($el.attr('value') || 'checked') : 'unchecked';
          break;
        case 'select':
          fieldValue = $el.find('option:selected').text() || $el.find('option').first().text();
          break;
        default:
          fieldValue = $el.attr('value') || $el.text().trim();
      }
      
      if (fieldName) {
        fields.push({
          fieldName: fieldName,
          fieldValue: fieldValue,
          fieldType: fieldType
        });
      }
    });
    
    // Find label-input pairs
    $form.find('label').each((_: any, element: any) => {
      const $label = $(element);
      const forAttr = $label.attr('for');
      const labelText = $label.text().trim();
      
      if (forAttr && labelText) {
        const $input = $form.find(`#${forAttr}`);
        if ($input.length > 0) {
          const existingField = fields.find(f => f.fieldName === labelText);
          if (!existingField) {
            const fieldType = $input.attr('type') || $input.prop('tagName')?.toLowerCase();
            let fieldValue = '';
            
            switch (fieldType) {
              case 'checkbox':
              case 'radio':
                fieldValue = $input.prop('checked') ? ($input.attr('value') || 'checked') : 'unchecked';
                break;
              default:
                fieldValue = $input.attr('value') || $input.text().trim();
            }
            
            fields.push({
              fieldName: labelText,
              fieldValue: fieldValue,
              fieldType: fieldType
            });
          }
        }
      }
    });
    
    return fields;
  }
  
  /**
   * Extract links with metadata
   */
  extractLinks($: any, $el: any): Link[] {
    const links: Link[] = [];
    
    $el.find('a[href]').each((_: any, element: any) => {
      const $link = $(element);
      const href = $link.attr('href');
      const text = $link.text().trim();
      
      if (href && text) {
        links.push({
          text: text,
          url: href,
          startIndex: 0 // Would need more complex text analysis to determine actual position
        });
      }
    });
    
    return links;
  }
  
  /**
   * Parse address components
   */
  parseAddress(text: string): any {
    const components: any = {};
    
    // Street address pattern
    const streetMatch = text.match(/(\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl))/i);
    if (streetMatch) {
      components.street = streetMatch[1].trim();
    }
    
    // ZIP code pattern
    const zipMatch = text.match(/\b(\d{5}(?:-\d{4})?)\b/);
    if (zipMatch) {
      components.zipCode = zipMatch[1];
    }
    
    // State + ZIP pattern
    const stateZipMatch = text.match(/\b([A-Z]{2})\s+(\d{5}(?:-\d{4})?)\b/);
    if (stateZipMatch) {
      components.state = stateZipMatch[1];
      components.zipCode = stateZipMatch[2];
    }
    
    // City pattern (before state/zip)
    const cityMatch = text.match(/,\s*([A-Za-z\s]+),?\s*[A-Z]{2}\s*\d{5}/);
    if (cityMatch) {
      components.city = cityMatch[1].trim();
    }
    
    return Object.keys(components).length > 0 ? components : undefined;
  }
  
  /**
   * Extract email addresses from text
   */
  extractEmailAddresses(text: string): string[] {
    const emailPattern = /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/g;
    return text.match(emailPattern) || [];
  }
  
  /**
   * Detect programming language in code blocks
   */
  detectCodeLanguage($el: any): string | undefined {
    // Check class attributes for language hints
    const className = $el.attr('class') || '';
    
    // Common language class patterns
    const languagePatterns = [
      { pattern: /language-(\w+)/i, group: 1 },
      { pattern: /lang-(\w+)/i, group: 1 },
      { pattern: /highlight-(\w+)/i, group: 1 },
      { pattern: /\b(javascript|js|typescript|ts|python|java|cpp|c\+\+|csharp|c#|php|ruby|go|rust|swift|kotlin|scala|html|css|sql|bash|shell|powershell|yaml|json|xml)\b/i, group: 1 }
    ];
    
    for (const { pattern, group } of languagePatterns) {
      const match = className.match(pattern);
      if (match && match[group]) {
        return match[group].toLowerCase();
      }
    }
    
    // Check data attributes
    const dataLang = $el.attr('data-language') || $el.attr('data-lang');
    if (dataLang) {
      return dataLang.toLowerCase();
    }
    
    // Try to detect from content patterns
    const text = $el.text();
    return this.detectLanguageFromContent(text);
  }
  
  /**
   * Detect programming language from code content
   */
  private detectLanguageFromContent(code: string): string | undefined {
    const languagePatterns: Array<{ language: string; patterns: RegExp[] }> = [
      { language: 'javascript', patterns: [/\b(function|const|let|var|=>|console\.log)\b/, /\$\(.*\)/, /require\(.*\)/] },
      { language: 'typescript', patterns: [/\b(interface|type|enum)\b/, /:\s*(string|number|boolean)/, /\bas\s+\w+/] },
      { language: 'python', patterns: [/\b(def|import|from|class|if __name__)\b/, /print\(/, /\bself\b/] },
      { language: 'java', patterns: [/\b(public|private|class|static|void)\b/, /System\.out\.println/, /\bString\[\]/] },
      { language: 'cpp', patterns: [/\b(#include|using namespace|std::)\b/, /cout\s*<</, /\bint main\b/] },
      { language: 'csharp', patterns: [/\b(using|namespace|public class)\b/, /Console\.WriteLine/, /\bstring\[\]/] },
      { language: 'php', patterns: [/<\?php/, /\$\w+/, /echo\s+/] },
      { language: 'ruby', patterns: [/\b(def|end|class|require)\b/, /puts\s+/, /\@\w+/] },
      { language: 'go', patterns: [/\b(package|import|func|var)\b/, /fmt\.Print/, /\bgo\s+\w+/] },
      { language: 'rust', patterns: [/\b(fn|let|mut|use|struct)\b/, /println!/, /\bSome\(|\bNone\b/] },
      { language: 'html', patterns: [/<\/?[a-z][\s\S]*>/i, /<!DOCTYPE/, /&\w+;/] },
      { language: 'css', patterns: [/\{[^}]*\}/, /\.[a-zA-Z][\w-]*\s*\{/, /@media\s+/] },
      { language: 'sql', patterns: [/\b(SELECT|FROM|WHERE|INSERT|UPDATE|DELETE)\b/i, /\bJOIN\b/i, /\bGROUP BY\b/i] },
      { language: 'bash', patterns: [/^#!/, /\$\w+/, /\becho\s+/, /\|\s*\w+/] },
      { language: 'json', patterns: [/^\s*\{[\s\S]*\}\s*$/, /"\w+":\s*/, /\[\s*\{/] },
      { language: 'xml', patterns: [/<\?xml/, /<\/\w+>/, /xmlns:/] },
      { language: 'yaml', patterns: [/^\s*\w+:\s*/, /^---/, /^\s*-\s+/] }
    ];
    
    for (const { language, patterns } of languagePatterns) {
      if (patterns.some((pattern: RegExp) => pattern.test(code))) {
        return language;
      }
    }
    
    return undefined;
  }
  
  /**
   * Extract mathematical formula type
   */
  detectFormulaType($el: any): 'latex' | 'mathml' | 'text' {
    const tagName = $el.prop('tagName')?.toLowerCase();
    const text = $el.text();
    
    // MathML elements
    if (tagName && ['math', 'mrow', 'mi', 'mn', 'mo'].includes(tagName)) {
      return 'mathml';
    }
    
    // LaTeX patterns
    if (/\$.*\$|\\[a-zA-Z]+\{.*\}|\\begin\{.*\}/.test(text)) {
      return 'latex';
    }
    
    return 'text';
  }
  
  /**
   * Extract coordinates from element positioning
   */
  extractCoordinates($el: any): CoordinatesMetadata | undefined {
    // This would typically require access to computed styles or layout information
    // For now, we'll extract basic positioning if available
    
    const style = $el.attr('style') || '';
    const position: Partial<{ top: number; left: number; width: number; height: number }> = {};
    
    // Extract CSS positioning
    const topMatch = style.match(/top:\s*(\d+)px/);
    const leftMatch = style.match(/left:\s*(\d+)px/);
    const widthMatch = style.match(/width:\s*(\d+)px/);
    const heightMatch = style.match(/height:\s*(\d+)px/);
    
    if (topMatch) position.top = parseInt(topMatch[1]);
    if (leftMatch) position.left = parseInt(leftMatch[1]);
    if (widthMatch) position.width = parseInt(widthMatch[1]);
    if (heightMatch) position.height = parseInt(heightMatch[1]);
    
    // If we have positioning information, create coordinate metadata
    if (position.top !== undefined && position.left !== undefined) {
      const points: Point[] = [
        { x: position.left, y: position.top },
        { x: position.left + (position.width || 0), y: position.top },
        { x: position.left + (position.width || 0), y: position.top + (position.height || 0) },
        { x: position.left, y: position.top + (position.height || 0) }
      ];
      
      return {
        points,
        system: {
          width: 1920, // Default viewport width
          height: 1080, // Default viewport height
          coordinateUnit: 'pixels'
        }
      };
    }
    
    return undefined;
  }
  
  /**
   * Extract emphasized text and their tags
   */
  extractEmphasis($: any, $el: any): { contents: string[]; tags: string[] } {
    const contents: string[] = [];
    const tags: string[] = [];
    
    const emphasisTags = ['strong', 'b', 'em', 'i', 'u', 'mark', 'ins', 'del'];
    
    emphasisTags.forEach(tag => {
      $el.find(tag).each((_: any, element: any) => {
        const $emphEl = $(element);
        const text = $emphEl.text().trim();
        if (text) {
          contents.push(text);
          tags.push(tag);
        }
      });
    });
    
    return { contents, tags };
  }
  
  /**
   * Detect if text contains page break indicators
   */
  isPageBreak($el: any): boolean {
    const tagName = $el.prop('tagName')?.toLowerCase();
    const className = $el.attr('class') || '';
    const style = $el.attr('style') || '';
    
    // HR tags are often page breaks
    if (tagName === 'hr') {
      return true;
    }
    
    // CSS page break indicators
    const pageBreakClasses = /\b(page-break|pagebreak|new-page)\b/i;
    const pageBreakStyles = /page-break-(before|after):\s*(always|page)/i;
    
    return pageBreakClasses.test(className) || pageBreakStyles.test(style);
  }
  
  /**
   * Extract table structure with headers and data
   */
  extractTableStructure($: any, $table: any): { headers?: string[]; rows: string[][]; } {
    const headers: string[] = [];
    const rows: string[][] = [];
    
    // Extract headers from thead or first row with th elements
    const $thead = $table.find('thead');
    if ($thead.length > 0) {
      $thead.find('tr').first().find('th, td').each((_: any, cell: any) => {
          headers.push($(cell).text().trim());      });
    } else {
      // Check if first row has th elements
      const $firstRow = $table.find('tr').first();
      const $thCells = $firstRow.find('th');
      if ($thCells.length > 0) {
        $thCells.each((_: any, cell: any) => {
        headers.push($(cell).text().trim());        });
      }
    }
    
    // Extract data rows
    const $dataRows = headers.length > 0 
      ? $table.find('tbody tr, tr').not($table.find('thead tr'))
      : $table.find('tr');
    
    $dataRows.each((_: any, row: any) => {
      const $row = $(row);
      const rowData: string[] = [];
      
      $row.find('td, th').each((_: any, cell: any) => {
        rowData.push($(cell).text().trim());
      });
      
      if (rowData.length > 0) {
        rows.push(rowData);
      }
    });
    
    // If no explicit headers found but we have data, use first row as headers
    if (headers.length === 0 && rows.length > 0) {
      headers.push(...rows.shift()!);
    }
    
    return {
      headers: headers.length > 0 ? headers : undefined,
      rows
    };
  }
}