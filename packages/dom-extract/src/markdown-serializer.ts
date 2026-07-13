/**
 * Markdown serializer for extracted data elements
 * Converts structured elements into LLM-friendly markdown format
 */

import type {
  Element,
  AnyElement,
  TableElement,
  ImageElement,
  FormElement,
  CheckBoxElement,
  RadioButtonElement,
  LinkElement,
  CodeElement,
  FormulaElement,
  CompositeElement,
  PartitionResult
} from './types';
import { ElementType } from './types';

export interface MarkdownSerializerOptions {
  includeMetadata?: boolean;
  includeElementIds?: boolean;
  includeCoordinates?: boolean;
  includePageNumbers?: boolean;
  preserveHierarchy?: boolean;
  maxTableWidth?: number;
  escapeSpecialChars?: boolean;
  includeFormFields?: boolean;
  includeImageMetadata?: boolean;
  customElementHandlers?: Partial<Record<ElementType, (element: AnyElement) => string>>;
}

export class MarkdownSerializer {
  private options: Required<MarkdownSerializerOptions>;

  constructor(options: MarkdownSerializerOptions = {}) {
    this.options = {
      includeMetadata: false,
      includeElementIds: false,
      includeCoordinates: false,
      includePageNumbers: true,
      preserveHierarchy: true,
      maxTableWidth: 120,
      escapeSpecialChars: true,
      includeFormFields: true,
      includeImageMetadata: true,
      customElementHandlers: {},
      ...options
    };
  }

  serialize(result: PartitionResult): string {
    const sections: string[] = [];

    if (this.options.includeMetadata && result.metadata) {
      sections.push(this.serializeDocumentMetadata(result.metadata));
    }

    const content = this.serializeElements(result.elements);
    if (content.trim()) {
      sections.push(content);
    }

    return sections.join('\n\n');
  }

  serializeElements(elements: Element[]): string {
    const sections: string[] = [];
    let currentPage: number | undefined;

    for (const element of elements) {
      if (this.options.includePageNumbers && 
          element.metadata.pageNumber !== undefined && 
          element.metadata.pageNumber !== currentPage) {
        currentPage = element.metadata.pageNumber;
        sections.push(`\n---\n**Page ${currentPage}**\n`);
      }

      const serialized = this.serializeElement(element);
      if (serialized.trim()) {
        sections.push(serialized);
      }
    }

    return sections.join('\n\n');
  }

  private serializeElement(element: AnyElement): string {
    const customHandler = this.options.customElementHandlers[element.type];
    if (customHandler) {
      return customHandler(element);
    }

    switch (element.type) {
      case ElementType.TITLE:
        return this.serializeTitle(element);
      case ElementType.HEADER:
      case ElementType.SECTION_HEADER:
        return this.serializeHeader(element);
      case ElementType.HEADLINE:
        return this.serializeHeadline(element);
      case ElementType.SUB_HEADLINE:
        return this.serializeSubHeadline(element);
      case ElementType.NARRATIVE_TEXT:
      case ElementType.TEXT:
      case ElementType.PARAGRAPH:
        return this.serializeText(element);
      case ElementType.LIST:
      case ElementType.LIST_ITEM:
      case ElementType.BULLETED_TEXT:
        return this.serializeListItem(element);
      case ElementType.TABLE:
        return this.serializeTable(element as TableElement);
      case ElementType.IMAGE:
      case ElementType.PICTURE:
      case ElementType.FIGURE:
        return this.serializeImage(element as ImageElement);
      case ElementType.FORM:
      case ElementType.FORM_KEYS_VALUES:
        return this.serializeForm(element as FormElement);
      case ElementType.CHECK_BOX_CHECKED:
      case ElementType.CHECK_BOX_UNCHECKED:
        return this.serializeCheckBox(element as CheckBoxElement);
      case ElementType.RADIO_BUTTON_CHECKED:
      case ElementType.RADIO_BUTTON_UNCHECKED:
        return this.serializeRadioButton(element as RadioButtonElement);
      case ElementType.LINK:
        return this.serializeLink(element as LinkElement);
      // ADDRESS and EMAIL_ADDRESS cases removed - caused false positives
      case ElementType.CODE_SNIPPET:
        return this.serializeCode(element as CodeElement);
      case ElementType.FORMULA:
        return this.serializeFormula(element as FormulaElement);
      case ElementType.COMPOSITE_ELEMENT:
        return this.serializeComposite(element as CompositeElement);
      case ElementType.CAPTION:
      case ElementType.FIGURE_CAPTION:
        return this.serializeCaption(element);
      case ElementType.FOOTNOTE:
        return this.serializeFootnote(element);
      case ElementType.ABSTRACT:
        return this.serializeAbstract(element);
      case ElementType.PAGE_BREAK:
        return '---';
      case ElementType.FIELD_NAME:
        return `**${this.escapeText(element.text)}:**`;
      case ElementType.VALUE:
        return this.escapeText(element.text);
      default:
        return this.serializeGeneric(element);
    }
  }

  private serializeTitle(element: Element): string {
    const level = this.getTitleLevel(element);
    const prefix = '#'.repeat(level);
    return `${prefix} ${this.escapeText(element.text)}${this.getElementSuffix(element)}`;
  }

  private serializeHeader(element: Element): string {
    return `## ${this.escapeText(element.text)}${this.getElementSuffix(element)}`;
  }

  private serializeHeadline(element: Element): string {
    return `### ${this.escapeText(element.text)}${this.getElementSuffix(element)}`;
  }

  private serializeSubHeadline(element: Element): string {
    return `#### ${this.escapeText(element.text)}${this.getElementSuffix(element)}`;
  }

  private serializeText(element: Element): string {
    return `${this.escapeText(element.text)}${this.getElementSuffix(element)}`;
  }

  private serializeListItem(element: Element): string {
    return `- ${this.escapeText(element.text)}${this.getElementSuffix(element)}`;
  }

  private serializeTable(element: TableElement): string {
    if (!element.rows || element.rows.length === 0) {
      return `**Table:** ${this.escapeText(element.text)}${this.getElementSuffix(element)}`;
    }

    const lines: string[] = [];
    
    if (element.headers && element.headers.length > 0) {
      const headerRow = element.headers.map(h => this.escapeText(h)).join(' | ');
      const separator = element.headers.map(() => '---').join(' | ');
      lines.push(`| ${headerRow} |`);
      lines.push(`| ${separator} |`);
    }

    for (const row of element.rows) {
      const rowText = row.map(cell => this.escapeText(String(cell))).join(' | ');
      lines.push(`| ${rowText} |`);
    }

    return lines.join('\n') + this.getElementSuffix(element);
  }

  private serializeImage(element: ImageElement): string {
    const alt = element.alt || element.text || 'Image';
    const src = element.src || element.metadata.imageUrl || '';
    
    let result = `![${this.escapeText(alt)}](${src})`;
    
    if (this.options.includeImageMetadata) {
      const metadata: string[] = [];
      if (element.width) metadata.push(`Width: ${element.width}px`);
      if (element.height) metadata.push(`Height: ${element.height}px`);
      if (element.metadata.imageMimeType) metadata.push(`Type: ${element.metadata.imageMimeType}`);
      
      if (metadata.length > 0) {
        result += `\n*${metadata.join(', ')}*`;
      }
    }
    
    return result + this.getElementSuffix(element);
  }

  private serializeForm(element: FormElement): string {
    if (!this.options.includeFormFields || !element.fields) {
      return `**Form:** ${this.escapeText(element.text)}${this.getElementSuffix(element)}`;
    }

    const lines: string[] = ['**Form:**'];
    for (const field of element.fields) {
      lines.push(`- **${this.escapeText(field.fieldName)}:** ${this.escapeText(field.fieldValue)}`);
    }
    
    return lines.join('\n') + this.getElementSuffix(element);
  }

  private serializeCheckBox(element: CheckBoxElement): string {
    const checked = element.checked ? '[x]' : '[ ]';
    const value = element.value ? ` ${this.escapeText(element.value)}` : '';
    return `${checked}${value}${this.getElementSuffix(element)}`;
  }

  private serializeRadioButton(element: RadioButtonElement): string {
    const checked = element.checked ? '(•)' : '( )';
    const value = element.value ? ` ${this.escapeText(element.value)}` : '';
    const group = element.groupName ? ` [${element.groupName}]` : '';
    return `${checked}${value}${group}${this.getElementSuffix(element)}`;
  }

  private serializeLink(element: LinkElement): string {
    const text = element.linkText || element.text;
    return `[${this.escapeText(text)}](${element.url})${this.getElementSuffix(element)}`;
  }

  // serializeAddress and serializeEmailAddress methods removed - caused false positives

  private serializeCode(element: CodeElement): string {
    const language = element.language || '';
    return `\`\`\`${language}\n${element.codeBlock}\n\`\`\`${this.getElementSuffix(element)}`;
  }

  private serializeFormula(element: FormulaElement): string {
    if (element.formulaType === 'latex') {
      return `$$${element.formula}$$${this.getElementSuffix(element)}`;
    }
    return `**Formula:** \`${element.formula}\`${this.getElementSuffix(element)}`;
  }

  private serializeComposite(element: CompositeElement): string {
    const childElements = element.elements.map(el => this.serializeElement(el)).join('\n\n');
    return `**Composite Element:**\n\n${childElements}${this.getElementSuffix(element)}`;
  }

  private serializeCaption(element: Element): string {
    return `*${this.escapeText(element.text)}*${this.getElementSuffix(element)}`;
  }

  private serializeFootnote(element: Element): string {
    return `> ${this.escapeText(element.text)}${this.getElementSuffix(element)}`;
  }

  private serializeAbstract(element: Element): string {
    return `**Abstract:** ${this.escapeText(element.text)}${this.getElementSuffix(element)}`;
  }

  private serializeGeneric(element: Element): string {
    return `${this.escapeText(element.text)}${this.getElementSuffix(element)}`;
  }

  private serializeDocumentMetadata(metadata: any): string {
    const lines: string[] = ['# Document Metadata'];
    
    if (metadata.filename) lines.push(`**Filename:** ${metadata.filename}`);
    if (metadata.filetype) lines.push(`**File Type:** ${metadata.filetype}`);
    if (metadata.pageCount) lines.push(`**Pages:** ${metadata.pageCount}`);
    if (metadata.totalElements) lines.push(`**Total Elements:** ${metadata.totalElements}`);
    if (metadata.processingTime) lines.push(`**Processing Time:** ${metadata.processingTime}ms`);
    
    if (metadata.elementTypeCounts) {
      lines.push('**Element Types:**');
      for (const [type, count] of Object.entries(metadata.elementTypeCounts)) {
        lines.push(`- ${type}: ${count}`);
      }
    }
    
    return lines.join('\n');
  }

  private getTitleLevel(element: Element): number {
    if (element.metadata.categoryDepth !== undefined) {
      return Math.min(Math.max(element.metadata.categoryDepth + 1, 1), 6);
    }
    return 1;
  }

  private getElementSuffix(element: Element): string {
    const parts: string[] = [];
    
    if (this.options.includeElementIds) {
      parts.push(`<!-- id: ${element.id} -->`);
    }
    
    if (this.options.includeCoordinates && element.metadata.coordinates?.points) {
      const coords = element.metadata.coordinates.points;
      if (coords.length > 0) {
        parts.push(`<!-- coords: (${coords[0].x},${coords[0].y}) -->`);
      }
    }
    
    return parts.length > 0 ? ` ${parts.join(' ')}` : '';
  }

  private escapeText(text: string): string {
    if (!this.options.escapeSpecialChars) {
      return text;
    }
    
    return text
      .replace(/\\/g, '\\\\')
      .replace(/\*/g, '\\*')
      .replace(/_/g, '\\_')
      .replace(/`/g, '\\`')
      .replace(/\[/g, '\\[')
      .replace(/\]/g, '\\]')
      .replace(/\(/g, '\\(')
      .replace(/\)/g, '\\)')
      .replace(/!/g, '\\!')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }
}

export function serializeToMarkdown(
  result: PartitionResult, 
  options?: MarkdownSerializerOptions
): string {
  const serializer = new MarkdownSerializer(options);
  return serializer.serialize(result);
}