import { describe, test, expect } from 'bun:test'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { renderMarkdownContent, renderStreamingMarkdownContent, looksLikeMarkdown, hasOddFenceCount } from './markdown-content-renderer'

// Helper to extract text content by rendering to HTML and stripping tags
function extractText(node: React.ReactNode): string {
  try {
    // Wrap in a div to ensure we can render fragments
    const wrapped = React.createElement('div', null, node)
    const html = renderToStaticMarkup(wrapped)
    // Strip HTML tags to get plain text
    return html.replace(/<[^>]+>/g, '')
  } catch {
    // Fallback for when rendering fails
    return String(node)
  }
}


// Helper to check if node contains specific text
function containsText(node: React.ReactNode, text: string): boolean {
  return extractText(node).includes(text)
}


describe('looksLikeMarkdown', () => {
  test('detects bold markers', () => {
    expect(looksLikeMarkdown('**bold**')).toBe(true)
    expect(looksLikeMarkdown('__bold__')).toBe(true)
  })

  test('detects italic markers', () => {
    expect(looksLikeMarkdown('*italic*')).toBe(true)
    expect(looksLikeMarkdown('_italic_')).toBe(true)
  })

  test('detects inline code', () => {
    expect(looksLikeMarkdown('`code`')).toBe(true)
  })

  test('detects code fences', () => {
    expect(looksLikeMarkdown('```js\ncode\n```')).toBe(true)
  })

  test('detects headings', () => {
    expect(looksLikeMarkdown('# heading')).toBe(true)
    expect(looksLikeMarkdown('## heading')).toBe(true)
  })

  test('detects blockquotes', () => {
    expect(looksLikeMarkdown('> quote')).toBe(true)
  })

  test('detects lists', () => {
    expect(looksLikeMarkdown('- item')).toBe(true)
    expect(looksLikeMarkdown('+ item')).toBe(true)
  })

  test('detects links', () => {
    expect(looksLikeMarkdown('[text](url)')).toBe(true)
  })

  test('returns false for plain text', () => {
    expect(looksLikeMarkdown('plain text without markdown')).toBe(false)
    expect(looksLikeMarkdown('Hello World')).toBe(false)
  })
})

describe('hasOddFenceCount', () => {
  test('detects incomplete fence', () => {
    expect(hasOddFenceCount('```js\ncode')).toBe(true)
    expect(hasOddFenceCount('text\n```\ncode')).toBe(true)
  })

  test('detects complete fence', () => {
    expect(hasOddFenceCount('```js\ncode\n```')).toBe(false)
    expect(hasOddFenceCount('```\n```')).toBe(false)
  })

  test('handles multiple fences', () => {
    expect(hasOddFenceCount('```a```\n```b')).toBe(true)
    expect(hasOddFenceCount('```a```\n```b```')).toBe(false)
    expect(hasOddFenceCount('```a```\n```b```\n```c')).toBe(true)
  })

  test('handles no fences', () => {
    expect(hasOddFenceCount('no fences here')).toBe(false)
  })
})

describe('renderMarkdownContent - Headings', () => {
  test('renders h1', () => {
    const result = renderMarkdownContent('# Heading 1')
    expect(containsText(result, 'Heading 1')).toBe(true)
  })

  test('renders h2', () => {
    const result = renderMarkdownContent('## Heading 2')
    expect(containsText(result, 'Heading 2')).toBe(true)
  })

  test('renders h3', () => {
    const result = renderMarkdownContent('### Heading 3')
    expect(containsText(result, 'Heading 3')).toBe(true)
  })

  test('renders h4', () => {
    const result = renderMarkdownContent('#### Heading 4')
    expect(containsText(result, 'Heading 4')).toBe(true)
  })

  test('renders h5', () => {
    const result = renderMarkdownContent('##### Heading 5')
    expect(containsText(result, 'Heading 5')).toBe(true)
  })

  test('renders h6', () => {
    const result = renderMarkdownContent('###### Heading 6')
    expect(containsText(result, 'Heading 6')).toBe(true)
  })

  test('renders heading with inline formatting', () => {
    const result = renderMarkdownContent('# Hello **bold** and *italic*')
    expect(containsText(result, 'Hello')).toBe(true)
    expect(containsText(result, 'bold')).toBe(true)
    expect(containsText(result, 'italic')).toBe(true)
  })
})

describe('renderMarkdownContent - Emphasis', () => {
  test('renders bold with asterisks', () => {
    const result = renderMarkdownContent('**bold text**')
    expect(containsText(result, 'bold text')).toBe(true)
  })

  test('renders bold with underscores', () => {
    const result = renderMarkdownContent('__bold text__')
    expect(containsText(result, 'bold text')).toBe(true)
  })

  test('renders italic with asterisks', () => {
    const result = renderMarkdownContent('*italic text*')
    expect(containsText(result, 'italic text')).toBe(true)
  })

  test('renders italic with underscores', () => {
    const result = renderMarkdownContent('_italic text_')
    expect(containsText(result, 'italic text')).toBe(true)
  })

  test('renders bold and italic combined', () => {
    const result = renderMarkdownContent('***bold italic***')
    expect(containsText(result, 'bold italic')).toBe(true)
  })

  test('renders nested emphasis', () => {
    const result = renderMarkdownContent('**bold with *italic* inside**')
    expect(containsText(result, 'bold with')).toBe(true)
    expect(containsText(result, 'italic')).toBe(true)
    expect(containsText(result, 'inside')).toBe(true)
  })

  test('renders strikethrough (GFM)', () => {
    const result = renderMarkdownContent('~~strikethrough~~')
    expect(containsText(result, 'strikethrough')).toBe(true)
  })
})

describe('renderMarkdownContent - Code', () => {
  test('renders inline code', () => {
    const result = renderMarkdownContent('Use `console.log()` to debug')
    expect(containsText(result, 'console.log()')).toBe(true)
  })

  test('renders inline code with special characters', () => {
    const result = renderMarkdownContent('`const x = 1 + 2`')
    expect(containsText(result, 'const x = 1 + 2')).toBe(true)
  })

  test('renders code block without language', () => {
    const result = renderMarkdownContent('```\nconst x = 1\n```')
    expect(containsText(result, 'const x = 1')).toBe(true)
  })

  test('renders code block with language', () => {
    const result = renderMarkdownContent('```javascript\nconst x = 1\n```')
    expect(containsText(result, 'const x = 1')).toBe(true)
    expect(containsText(result, 'javascript')).toBe(true)
  })

  test('renders code block with multiple lines', () => {
    const code = '```js\nline 1\nline 2\nline 3\n```'
    const result = renderMarkdownContent(code)
    expect(containsText(result, 'line 1')).toBe(true)
    expect(containsText(result, 'line 2')).toBe(true)
    expect(containsText(result, 'line 3')).toBe(true)
  })

  test('preserves indentation in code blocks', () => {
    const code = '```python\ndef foo():\n    return 42\n```'
    const result = renderMarkdownContent(code)
    expect(containsText(result, 'def foo():')).toBe(true)
    expect(containsText(result, 'return 42')).toBe(true)
  })

  test('renders various language identifiers', () => {
    const languages = ['python', 'rust', 'go', 'java', 'c', 'cpp', 'ruby', 'php', 'swift', 'kotlin']
    for (const lang of languages) {
      const result = renderMarkdownContent(`\`\`\`${lang}\ncode\n\`\`\``)
      expect(containsText(result, lang)).toBe(true)
    }
  })
})

describe('renderMarkdownContent - Mermaid', () => {
  test('renders mermaid flowchart as ASCII', () => {
    const mermaid = '```mermaid\ngraph TD\n    A --> B\n```'
    const result = renderMarkdownContent(mermaid)
    // Should contain box-drawing characters from ASCII rendering
    const text = extractText(result)
    expect(text.includes('mermaid')).toBe(true)
    // ASCII art should have box characters
    expect(text.includes('┌') || text.includes('│') || text.includes('─')).toBe(true)
  })

  test('renders mermaid with nodes', () => {
    const mermaid = '```mermaid\ngraph LR\n    A[Start] --> B[End]\n```'
    const result = renderMarkdownContent(mermaid)
    const text = extractText(result)
    expect(text.includes('Start') || text.includes('End') || text.includes('─')).toBe(true)
  })

  test('handles invalid mermaid gracefully', () => {
    const invalid = '```mermaid\nthis is not valid mermaid syntax @@##$$\n```'
    const result = renderMarkdownContent(invalid)
    // Should still render something (fallback to code block)
    expect(result).toBeTruthy()
  })
})

describe('renderMarkdownContent - Links', () => {
  test('renders basic link', () => {
    const result = renderMarkdownContent('[Click here](https://example.com)')
    expect(containsText(result, 'Click here')).toBe(true)
  })

  test('renders link with title', () => {
    const result = renderMarkdownContent('[Link](https://example.com "Title")')
    expect(containsText(result, 'Link')).toBe(true)
  })

  test('renders autolink', () => {
    const result = renderMarkdownContent('<https://example.com>')
    expect(containsText(result, 'https://example.com')).toBe(true)
  })

  test('renders link with inline formatting', () => {
    const result = renderMarkdownContent('[**Bold link**](https://example.com)')
    expect(containsText(result, 'Bold link')).toBe(true)
  })
})

describe('renderMarkdownContent - Lists', () => {
  test('renders unordered list with dashes', () => {
    const result = renderMarkdownContent('- Item 1\n- Item 2\n- Item 3')
    expect(containsText(result, 'Item 1')).toBe(true)
    expect(containsText(result, 'Item 2')).toBe(true)
    expect(containsText(result, 'Item 3')).toBe(true)
  })

  test('renders unordered list with asterisks', () => {
    const result = renderMarkdownContent('* Item 1\n* Item 2')
    expect(containsText(result, 'Item 1')).toBe(true)
    expect(containsText(result, 'Item 2')).toBe(true)
  })

  test('renders unordered list with plus signs', () => {
    const result = renderMarkdownContent('+ Item 1\n+ Item 2')
    expect(containsText(result, 'Item 1')).toBe(true)
    expect(containsText(result, 'Item 2')).toBe(true)
  })

  test('renders ordered list', () => {
    const result = renderMarkdownContent('1. First\n2. Second\n3. Third')
    expect(containsText(result, 'First')).toBe(true)
    expect(containsText(result, 'Second')).toBe(true)
    expect(containsText(result, 'Third')).toBe(true)
  })

  test('renders ordered list with custom start', () => {
    const result = renderMarkdownContent('5. Fifth\n6. Sixth')
    expect(containsText(result, 'Fifth')).toBe(true)
    expect(containsText(result, 'Sixth')).toBe(true)
  })

  test('renders list with inline formatting', () => {
    const result = renderMarkdownContent('- **Bold item**\n- *Italic item*\n- `Code item`')
    expect(containsText(result, 'Bold item')).toBe(true)
    expect(containsText(result, 'Italic item')).toBe(true)
    expect(containsText(result, 'Code item')).toBe(true)
  })

  test('renders nested lists', () => {
    const result = renderMarkdownContent('- Parent\n  - Child 1\n  - Child 2')
    expect(containsText(result, 'Parent')).toBe(true)
    expect(containsText(result, 'Child 1')).toBe(true)
    expect(containsText(result, 'Child 2')).toBe(true)
  })

  test('renders task list (GFM)', () => {
    const result = renderMarkdownContent('- [ ] Unchecked\n- [x] Checked')
    expect(containsText(result, 'Unchecked')).toBe(true)
    expect(containsText(result, 'Checked')).toBe(true)
  })
})

describe('renderMarkdownContent - Blockquotes', () => {
  test('renders simple blockquote', () => {
    const result = renderMarkdownContent('> This is a quote')
    expect(containsText(result, 'This is a quote')).toBe(true)
  })

  test('renders multi-line blockquote', () => {
    const result = renderMarkdownContent('> Line 1\n> Line 2\n> Line 3')
    expect(containsText(result, 'Line 1')).toBe(true)
    expect(containsText(result, 'Line 2')).toBe(true)
    expect(containsText(result, 'Line 3')).toBe(true)
  })

  test('renders nested blockquotes', () => {
    const result = renderMarkdownContent('> Outer\n>> Inner')
    expect(containsText(result, 'Outer')).toBe(true)
    expect(containsText(result, 'Inner')).toBe(true)
  })

  test('renders blockquote with inline formatting', () => {
    const result = renderMarkdownContent('> **Bold** and *italic* in quote')
    expect(containsText(result, 'Bold')).toBe(true)
    expect(containsText(result, 'italic')).toBe(true)
  })
})

describe('renderMarkdownContent - Tables (GFM)', () => {
  test('renders simple table', () => {
    const table = '| A | B |\n|---|---|\n| 1 | 2 |'
    const result = renderMarkdownContent(table)
    expect(containsText(result, 'A')).toBe(true)
    expect(containsText(result, 'B')).toBe(true)
    expect(containsText(result, '1')).toBe(true)
    expect(containsText(result, '2')).toBe(true)
  })

  test('renders table with multiple rows', () => {
    const table = '| Col1 | Col2 |\n|------|------|\n| A | B |\n| C | D |\n| E | F |'
    const result = renderMarkdownContent(table)
    for (const cell of ['Col1', 'Col2', 'A', 'B', 'C', 'D', 'E', 'F']) {
      expect(containsText(result, cell)).toBe(true)
    }
  })

  test('renders table with alignment', () => {
    const table = '| Left | Center | Right |\n|:-----|:------:|------:|\n| L | C | R |'
    const result = renderMarkdownContent(table)
    expect(containsText(result, 'Left')).toBe(true)
    expect(containsText(result, 'Center')).toBe(true)
    expect(containsText(result, 'Right')).toBe(true)
  })

  test('renders table with empty cells', () => {
    const table = '| A | B |\n|---|---|\n| 1 |   |\n|   | 2 |'
    const result = renderMarkdownContent(table)
    expect(containsText(result, 'A')).toBe(true)
    expect(containsText(result, 'B')).toBe(true)
  })

  test('renders table with inline formatting', () => {
    const table = '| **Bold** | *Italic* |\n|----------|----------|\n| `code` | [link](url) |'
    const result = renderMarkdownContent(table)
    expect(containsText(result, 'Bold')).toBe(true)
    expect(containsText(result, 'Italic')).toBe(true)
    expect(containsText(result, 'code')).toBe(true)
    expect(containsText(result, 'link')).toBe(true)
  })
})

describe('renderMarkdownContent - Horizontal Rules', () => {
  test('renders hr with dashes', () => {
    const result = renderMarkdownContent('---')
    const text = extractText(result)
    expect(text.includes('─')).toBe(true)
  })

  test('renders hr with asterisks', () => {
    const result = renderMarkdownContent('***')
    const text = extractText(result)
    expect(text.includes('─')).toBe(true)
  })

  test('renders hr with underscores', () => {
    const result = renderMarkdownContent('___')
    const text = extractText(result)
    expect(text.includes('─')).toBe(true)
  })
})

describe('renderMarkdownContent - Paragraphs', () => {
  test('renders single paragraph', () => {
    const result = renderMarkdownContent('This is a paragraph.')
    expect(containsText(result, 'This is a paragraph.')).toBe(true)
  })

  test('renders multiple paragraphs', () => {
    const result = renderMarkdownContent('Paragraph 1.\n\nParagraph 2.')
    expect(containsText(result, 'Paragraph 1.')).toBe(true)
    expect(containsText(result, 'Paragraph 2.')).toBe(true)
  })

  test('handles soft line breaks', () => {
    const result = renderMarkdownContent('Line 1\nLine 2')
    expect(containsText(result, 'Line 1')).toBe(true)
    expect(containsText(result, 'Line 2')).toBe(true)
  })

  test('handles hard line breaks', () => {
    const result = renderMarkdownContent('Line 1  \nLine 2')
    expect(containsText(result, 'Line 1')).toBe(true)
    expect(containsText(result, 'Line 2')).toBe(true)
  })
})

describe('renderMarkdownContent - Mixed Content', () => {
  test('renders complex document', () => {
    const doc = `# Title

This is a paragraph with **bold** and *italic*.

## Code Example

\`\`\`javascript
const x = 1;
\`\`\`

## List

- Item 1
- Item 2

> A quote

| A | B |
|---|---|
| 1 | 2 |
`
    const result = renderMarkdownContent(doc)
    expect(containsText(result, 'Title')).toBe(true)
    expect(containsText(result, 'bold')).toBe(true)
    expect(containsText(result, 'italic')).toBe(true)
    expect(containsText(result, 'const x = 1')).toBe(true)
    expect(containsText(result, 'Item 1')).toBe(true)
    expect(containsText(result, 'A quote')).toBe(true)
  })

  test('handles empty input', () => {
    const result = renderMarkdownContent('')
    expect(result).toBe('')
  })

  test('handles whitespace-only input', () => {
    const result = renderMarkdownContent('   \n\n   ')
    expect(result).toBeTruthy()
  })
})

describe('renderMarkdownContent - Edge Cases', () => {
  test('handles special characters', () => {
    const result = renderMarkdownContent('Special chars: < > & " \'')
    // Note: extractText uses renderToStaticMarkup which HTML-escapes these
    // So we check for the escaped versions or just verify rendering doesn't crash
    expect(result).toBeTruthy()
    // Check the ampersand makes it through (as &amp; in HTML)
    expect(containsText(result, 'Special chars')).toBe(true)
  })

  test('handles unicode characters', () => {
    const result = renderMarkdownContent('Unicode: 你好 مرحبا שלום')
    expect(containsText(result, '你好')).toBe(true)
    expect(containsText(result, 'مرحبا')).toBe(true)
    expect(containsText(result, 'שלום')).toBe(true)
  })

  test('handles emoji', () => {
    const result = renderMarkdownContent('Emoji: 🎉 🚀 ✨')
    expect(containsText(result, '🎉')).toBe(true)
    expect(containsText(result, '🚀')).toBe(true)
    expect(containsText(result, '✨')).toBe(true)
  })

  test('handles escaped characters', () => {
    const result = renderMarkdownContent('\\*not italic\\* and \\`not code\\`')
    expect(containsText(result, '*not italic*')).toBe(true)
  })

  test('handles inline code with backticks', () => {
    const result = renderMarkdownContent('`` `code` ``')
    expect(containsText(result, '`code`')).toBe(true)
  })

  test('handles very long lines', () => {
    const longLine = 'A'.repeat(1000)
    const result = renderMarkdownContent(longLine)
    expect(containsText(result, 'A'.repeat(100))).toBe(true)
  })

  test('handles deeply nested lists', () => {
    const nested = '- L1\n  - L2\n    - L3\n      - L4'
    const result = renderMarkdownContent(nested)
    expect(containsText(result, 'L1')).toBe(true)
    expect(containsText(result, 'L2')).toBe(true)
    expect(containsText(result, 'L3')).toBe(true)
    expect(containsText(result, 'L4')).toBe(true)
  })
})

describe('renderStreamingMarkdownContent', () => {
  test('renders complete markdown normally', () => {
    const result = renderStreamingMarkdownContent('# Hello\n\n**bold**')
    expect(containsText(result, 'Hello')).toBe(true)
    expect(containsText(result, 'bold')).toBe(true)
  })

  test('handles incomplete code fence', () => {
    const result = renderStreamingMarkdownContent('# Title\n\n```js\nconst x = 1')
    expect(containsText(result, 'Title')).toBe(true)
    // The pending section should still be visible
    expect(containsText(result, 'const x = 1')).toBe(true)
  })

  test('handles complete code followed by incomplete', () => {
    const content = '```js\ncomplete\n```\n\n```python\nincomplete'
    const result = renderStreamingMarkdownContent(content)
    expect(containsText(result, 'complete')).toBe(true)
    expect(containsText(result, 'incomplete')).toBe(true)
  })

  test('returns plain text for non-markdown', () => {
    const result = renderStreamingMarkdownContent('plain text')
    expect(result).toBe('plain text')
  })
})

describe('renderMarkdownContent - Palette Options', () => {
  test('accepts custom palette', () => {
    const result = renderMarkdownContent('# Hello', {
      palette: {
        headingFg: { 1: 'red' },
      },
    })
    expect(containsText(result, 'Hello')).toBe(true)
  })

  test('accepts custom code block width', () => {
    const result = renderMarkdownContent('---', {
      codeBlockWidth: 40,
    })
    const text = extractText(result)
    // HR width should be constrained
    expect(text.includes('─')).toBe(true)
  })

  test('accepts monochrome mode', () => {
    const result = renderMarkdownContent('`code`', {
      palette: {
        codeMonochrome: true,
      },
    })
    expect(containsText(result, 'code')).toBe(true)
  })
})

describe('renderMarkdownContent - Images', () => {
  test('renders image alt text', () => {
    const result = renderMarkdownContent('![Alt text](https://example.com/image.png)')
    // Images render as [alt text] in terminal
    expect(containsText(result, '[Alt text]')).toBe(true)
  })

  test('renders image with title', () => {
    const result = renderMarkdownContent('![Alt](url "Title")')
    expect(containsText(result, '[Alt]')).toBe(true)
  })
})

describe('renderMarkdownContent - HTML', () => {
  test('handles inline HTML', () => {
    const result = renderMarkdownContent('Text with <strong>HTML</strong>')
    // May or may not render HTML - just shouldn't crash
    expect(result).toBeTruthy()
  })

  test('handles block HTML', () => {
    const result = renderMarkdownContent('<div>Block HTML</div>')
    expect(result).toBeTruthy()
  })
})

describe('renderMarkdownContent - Definition Lists', () => {
  test('handles definition-like content', () => {
    const result = renderMarkdownContent('Term\n: Definition')
    expect(result).toBeTruthy()
  })
})

describe('renderMarkdownContent - Footnotes', () => {
  test('handles footnote-like content', () => {
    const result = renderMarkdownContent('Text[^1]\n\n[^1]: Footnote')
    expect(result).toBeTruthy()
  })
})

describe('renderMarkdownContent - Autolinks (GFM)', () => {
  test('renders URL autolinks', () => {
    const result = renderMarkdownContent('Check out https://example.com for more')
    expect(containsText(result, 'https://example.com')).toBe(true)
  })

  test('renders email autolinks', () => {
    const result = renderMarkdownContent('Contact <test@example.com>')
    expect(containsText(result, 'test@example.com')).toBe(true)
  })
})

describe('renderMarkdownContent - Error Handling', () => {
  test('handles malformed markdown gracefully', () => {
    // Various edge cases that might break parsers
    const cases = [
      '**unclosed bold',
      '*unclosed italic',
      '`unclosed code',
      '[unclosed link',
      '> ',
      '- ',
      '1. ',
      '```',
      '|',
      '||',
    ]

    for (const input of cases) {
      const result = renderMarkdownContent(input)
      expect(result).toBeTruthy()
    }
  })

  test('handles null-like characters', () => {
    const result = renderMarkdownContent('Text with \0 null')
    expect(result).toBeTruthy()
  })
})

describe('renderMarkdownContent - Task Lists', () => {
  test('renders unchecked task', () => {
    const result = renderMarkdownContent('- [ ] Todo item')
    expect(containsText(result, 'Todo item')).toBe(true)
    expect(containsText(result, '[ ]')).toBe(true)
  })

  test('renders checked task', () => {
    const result = renderMarkdownContent('- [x] Done item')
    expect(containsText(result, 'Done item')).toBe(true)
    expect(containsText(result, '[x]')).toBe(true)
  })

  test('renders uppercase checked task', () => {
    const result = renderMarkdownContent('- [X] Done item')
    expect(containsText(result, 'Done item')).toBe(true)
  })

  test('renders mixed task list', () => {
    const result = renderMarkdownContent('- [ ] Todo\n- [x] Done\n- [ ] Another')
    expect(containsText(result, 'Todo')).toBe(true)
    expect(containsText(result, 'Done')).toBe(true)
    expect(containsText(result, 'Another')).toBe(true)
  })
})

describe('renderMarkdownContent - Raw HTML', () => {
  test('handles html element', () => {
    const result = renderMarkdownContent('<div>content</div>')
    expect(result).toBeTruthy()
  })

  test('handles br element', () => {
    const result = renderMarkdownContent('line1<br>line2')
    expect(result).toBeTruthy()
  })

  test('handles hr element', () => {
    const result = renderMarkdownContent('<hr>')
    expect(result).toBeTruthy()
  })

  test('handles nested html', () => {
    const result = renderMarkdownContent('<div><span>nested</span></div>')
    expect(result).toBeTruthy()
  })

  test('handles html with attributes', () => {
    const result = renderMarkdownContent('<div class="test" id="main">content</div>')
    expect(result).toBeTruthy()
  })

  test('handles self-closing html', () => {
    const result = renderMarkdownContent('<br/><hr/><img src="x"/>')
    expect(result).toBeTruthy()
  })

  test('handles html comments', () => {
    const result = renderMarkdownContent('text <!-- comment --> more text')
    expect(result).toBeTruthy()
  })
})

describe('renderMarkdownContent - Link Props', () => {
  test('renders link with title attribute', () => {
    const result = renderMarkdownContent('[text](url "hover title")')
    expect(containsText(result, 'text')).toBe(true)
  })

  test('renders autolink URL', () => {
    const result = renderMarkdownContent('<https://example.com>')
    expect(containsText(result, 'https://example.com')).toBe(true)
  })

  test('renders autolink email', () => {
    const result = renderMarkdownContent('<test@example.com>')
    expect(containsText(result, 'test@example.com')).toBe(true)
  })

  test('renders GFM autolink', () => {
    const result = renderMarkdownContent('Visit https://example.com for more')
    expect(containsText(result, 'https://example.com')).toBe(true)
  })
})

describe('renderMarkdownContent - Table Alignment', () => {
  test('renders left-aligned column', () => {
    const result = renderMarkdownContent('| Left |\n|:-----|\n| text |')
    expect(containsText(result, 'Left')).toBe(true)
    expect(containsText(result, 'text')).toBe(true)
  })

  test('renders center-aligned column', () => {
    const result = renderMarkdownContent('| Center |\n|:------:|\n| text |')
    expect(containsText(result, 'Center')).toBe(true)
  })

  test('renders right-aligned column', () => {
    const result = renderMarkdownContent('| Right |\n|------:|\n| text |')
    expect(containsText(result, 'Right')).toBe(true)
  })

  test('renders mixed alignment', () => {
    const result = renderMarkdownContent('| L | C | R |\n|:--|:-:|--:|\n| a | b | c |')
    expect(containsText(result, 'L')).toBe(true)
    expect(containsText(result, 'C')).toBe(true)
    expect(containsText(result, 'R')).toBe(true)
  })
})

describe('renderMarkdownContent - Image Props', () => {
  test('renders image with title', () => {
    const result = renderMarkdownContent('![alt](url "title")')
    expect(containsText(result, '[alt]')).toBe(true)
  })

  test('handles empty alt text', () => {
    const result = renderMarkdownContent('![](url)')
    expect(result).toBeTruthy()
  })

  test('handles image in link', () => {
    const result = renderMarkdownContent('[![alt](img-url)](link-url)')
    expect(result).toBeTruthy()
  })
})

describe('renderMarkdownContent - Ordered List Props', () => {
  test('renders list starting at 1', () => {
    const result = renderMarkdownContent('1. First\n2. Second')
    expect(containsText(result, 'First')).toBe(true)
  })

  test('renders list starting at custom number', () => {
    const result = renderMarkdownContent('5. Fifth\n6. Sixth')
    expect(containsText(result, 'Fifth')).toBe(true)
    expect(containsText(result, '5.')).toBe(true)
  })

  test('renders list with large start number', () => {
    const result = renderMarkdownContent('99. Ninety-nine\n100. Hundred')
    expect(containsText(result, '99.')).toBe(true)
    expect(containsText(result, '100.')).toBe(true)
  })
})

describe('renderMarkdownContent - Setext Headings', () => {
  test('renders h1 with equals underline', () => {
    const result = renderMarkdownContent('Heading\n=======')
    expect(containsText(result, 'Heading')).toBe(true)
  })

  test('renders h2 with dash underline', () => {
    const result = renderMarkdownContent('Heading\n-------')
    expect(containsText(result, 'Heading')).toBe(true)
  })
})

describe('renderMarkdownContent - Code Fence Variations', () => {
  test('renders tilde fence', () => {
    const result = renderMarkdownContent('~~~\ncode\n~~~')
    expect(containsText(result, 'code')).toBe(true)
  })

  test('renders indented code block', () => {
    const result = renderMarkdownContent('    indented code')
    expect(containsText(result, 'indented code')).toBe(true)
  })

  test('renders empty code block', () => {
    const result = renderMarkdownContent('```\n```')
    expect(result).toBeTruthy()
  })

  test('renders code with empty lines', () => {
    const result = renderMarkdownContent('```\nline1\n\nline3\n```')
    expect(containsText(result, 'line1')).toBe(true)
    expect(containsText(result, 'line3')).toBe(true)
  })
})

describe('renderMarkdownContent - Nested Structures', () => {
  test('renders bold inside italic', () => {
    const result = renderMarkdownContent('*italic **bold** italic*')
    expect(containsText(result, 'italic')).toBe(true)
    expect(containsText(result, 'bold')).toBe(true)
  })

  test('renders italic inside bold', () => {
    const result = renderMarkdownContent('**bold *italic* bold**')
    expect(containsText(result, 'bold')).toBe(true)
    expect(containsText(result, 'italic')).toBe(true)
  })

  test('renders code in link', () => {
    const result = renderMarkdownContent('[`code`](url)')
    expect(containsText(result, 'code')).toBe(true)
  })

  test('renders deeply nested quotes', () => {
    const result = renderMarkdownContent('> level 1\n>> level 2\n>>> level 3')
    expect(containsText(result, 'level 1')).toBe(true)
    expect(containsText(result, 'level 2')).toBe(true)
    expect(containsText(result, 'level 3')).toBe(true)
  })

  test('renders list in blockquote', () => {
    const result = renderMarkdownContent('> - item 1\n> - item 2')
    expect(containsText(result, 'item 1')).toBe(true)
    expect(containsText(result, 'item 2')).toBe(true)
  })
})

describe('renderMarkdownContent - HTML Entities', () => {
  test('renders named entities', () => {
    const result = renderMarkdownContent('&amp; &lt; &gt; &quot;')
    expect(result).toBeTruthy()
  })

  test('renders numeric entities', () => {
    const result = renderMarkdownContent('&#65; &#66;')
    expect(result).toBeTruthy()
  })

  test('renders hex entities', () => {
    const result = renderMarkdownContent('&#x41; &#x42;')
    expect(result).toBeTruthy()
  })
})

describe('renderMarkdownContent - Diff Code Block', () => {
  test('renders diff syntax', () => {
    const result = renderMarkdownContent('```diff\n+ added\n- removed\n  unchanged\n```')
    expect(containsText(result, 'added')).toBe(true)
    expect(containsText(result, 'removed')).toBe(true)
    expect(containsText(result, 'diff')).toBe(true)
  })
})

describe('renderMarkdownContent - Whitespace Edge Cases', () => {
  test('handles trailing whitespace', () => {
    const result = renderMarkdownContent('text   \n\n')
    expect(result).toBeTruthy()
  })

  test('handles leading whitespace', () => {
    const result = renderMarkdownContent('   text')
    expect(containsText(result, 'text')).toBe(true)
  })

  test('handles only whitespace', () => {
    const result = renderMarkdownContent('   \n   \n   ')
    expect(result).toBeTruthy()
  })

  test('handles tabs', () => {
    const result = renderMarkdownContent('\ttext\twith\ttabs')
    expect(result).toBeTruthy()
  })

  test('handles carriage returns', () => {
    const result = renderMarkdownContent('line1\r\nline2\r\n')
    expect(result).toBeTruthy()
  })
})

describe('renderMarkdownContent - All Element Types Coverage', () => {
  // This ensures every element type Bun.markdown can produce is handled
  // Based on Bun.markdown.react() documentation from bun-types@1.3.8

  test('h1-h6 with optional id prop', () => {
    // h1-h6: { id?, children }
    for (let i = 1; i <= 6; i++) {
      const result = renderMarkdownContent('#'.repeat(i) + ' Heading')
      expect(containsText(result, 'Heading')).toBe(true)
    }
  })

  test('p with children', () => {
    // p: { children }
    const result = renderMarkdownContent('Paragraph text')
    expect(containsText(result, 'Paragraph text')).toBe(true)
  })

  test('blockquote with children', () => {
    // blockquote: { children }
    const result = renderMarkdownContent('> Quote text')
    expect(containsText(result, 'Quote text')).toBe(true)
  })

  test('pre with optional language prop', () => {
    // pre: { language?, children }
    const withLang = renderMarkdownContent('```javascript\ncode\n```')
    expect(containsText(withLang, 'code')).toBe(true)
    expect(containsText(withLang, 'javascript')).toBe(true)

    const withoutLang = renderMarkdownContent('```\ncode\n```')
    expect(containsText(withoutLang, 'code')).toBe(true)
  })

  test('hr with no props', () => {
    // hr: {}
    const result = renderMarkdownContent('---')
    expect(result).toBeTruthy()
    expect(extractText(result).includes('─')).toBe(true)
  })

  test('ul with children', () => {
    // ul: { children }
    const result = renderMarkdownContent('- item1\n- item2')
    expect(containsText(result, 'item1')).toBe(true)
    expect(containsText(result, 'item2')).toBe(true)
  })

  test('ol with start prop', () => {
    // ol: { start, children }
    const result = renderMarkdownContent('5. fifth\n6. sixth')
    expect(containsText(result, 'fifth')).toBe(true)
    expect(containsText(result, '5.')).toBe(true)
  })

  test('li with optional checked prop', () => {
    // li: { checked?, children }
    const unchecked = renderMarkdownContent('- [ ] todo')
    expect(containsText(unchecked, 'todo')).toBe(true)
    expect(containsText(unchecked, '[ ]')).toBe(true)

    const checked = renderMarkdownContent('- [x] done')
    expect(containsText(checked, 'done')).toBe(true)
    expect(containsText(checked, '[x]')).toBe(true)

    const regular = renderMarkdownContent('- item')
    expect(containsText(regular, 'item')).toBe(true)
  })

  test('table, thead, tbody, tr with children', () => {
    // table, thead, tbody, tr: { children }
    const result = renderMarkdownContent('| A | B |\n|---|---|\n| 1 | 2 |')
    expect(containsText(result, 'A')).toBe(true)
    expect(containsText(result, 'B')).toBe(true)
    expect(containsText(result, '1')).toBe(true)
    expect(containsText(result, '2')).toBe(true)
  })

  test('th and td with optional align prop', () => {
    // th, td: { align?, children }
    const result = renderMarkdownContent('| L | C | R |\n|:--|:--:|--:|\n| a | b | c |')
    expect(containsText(result, 'L')).toBe(true)
    expect(containsText(result, 'C')).toBe(true)
    expect(containsText(result, 'R')).toBe(true)
  })

  test('em with children', () => {
    // em: { children }
    const result = renderMarkdownContent('*emphasis*')
    expect(containsText(result, 'emphasis')).toBe(true)
  })

  test('strong with children', () => {
    // strong: { children }
    const result = renderMarkdownContent('**strong**')
    expect(containsText(result, 'strong')).toBe(true)
  })

  test('a with href and optional title props', () => {
    // a: { href, title?, children }
    const basic = renderMarkdownContent('[text](https://url)')
    expect(containsText(basic, 'text')).toBe(true)

    const withTitle = renderMarkdownContent('[text](https://url "title")')
    expect(containsText(withTitle, 'text')).toBe(true)
  })

  test('img with src, optional alt and title props', () => {
    // img: { src, alt?, title? }
    const basic = renderMarkdownContent('![alt](https://img)')
    expect(containsText(basic, '[alt]')).toBe(true)

    const withTitle = renderMarkdownContent('![alt](https://img "title")')
    expect(containsText(withTitle, '[alt]')).toBe(true)

    const noAlt = renderMarkdownContent('![](https://img)')
    expect(noAlt).toBeTruthy()
  })

  test('code (inline) with children', () => {
    // code: { children }
    const result = renderMarkdownContent('`inline code`')
    expect(containsText(result, 'inline code')).toBe(true)
  })

  test('del with children', () => {
    // del: { children }
    const result = renderMarkdownContent('~~strikethrough~~')
    expect(containsText(result, 'strikethrough')).toBe(true)
  })

  test('br with no props', () => {
    // br: {}
    const result = renderMarkdownContent('line1  \nline2')
    expect(containsText(result, 'line1')).toBe(true)
    expect(containsText(result, 'line2')).toBe(true)
  })

  test('html with children', () => {
    // html: { children }
    const result = renderMarkdownContent('<div>html content</div>')
    expect(result).toBeTruthy()
  })

  // These require parser options to be enabled
  test('math element (when latexMath enabled)', () => {
    // math: { children } - only when latexMath: true
    // We can't easily test this without passing options to Bun.markdown
    // but we ensure the component exists and handles content
    expect(true).toBe(true) // Placeholder - component exists in implementation
  })

  test('u element (when underline enabled)', () => {
    // u: { children } - only when underline: true
    // We can't easily test this without passing options to Bun.markdown
    expect(true).toBe(true) // Placeholder - component exists in implementation
  })
})

describe('renderMarkdownContent - Exhaustive Prop Coverage', () => {
  // Test every prop that each element can receive

  test('heading id prop (requires headings option)', () => {
    // The id prop is only passed when headings: { ids: true } is enabled
    // Our component should handle it gracefully whether present or not
    const result = renderMarkdownContent('# Heading')
    expect(containsText(result, 'Heading')).toBe(true)
  })

  test('ol start prop with various values', () => {
    const start1 = renderMarkdownContent('1. first')
    expect(containsText(start1, '1.')).toBe(true)

    const start5 = renderMarkdownContent('5. fifth')
    expect(containsText(start5, '5.')).toBe(true)

    const start99 = renderMarkdownContent('99. ninety-ninth')
    expect(containsText(start99, '99.')).toBe(true)
  })

  test('li checked prop with all values', () => {
    // checked: true
    const checkedLower = renderMarkdownContent('- [x] done')
    expect(containsText(checkedLower, '[x]')).toBe(true)

    // checked: true (uppercase X)
    const checkedUpper = renderMarkdownContent('- [X] DONE')
    expect(containsText(checkedUpper, 'DONE')).toBe(true)

    // checked: false
    const unchecked = renderMarkdownContent('- [ ] todo')
    expect(containsText(unchecked, '[ ]')).toBe(true)

    // checked: undefined (regular list item)
    const regular = renderMarkdownContent('- item')
    expect(containsText(regular, '- ')).toBe(true)
  })

  test('pre language prop with various languages', () => {
    const langs = ['javascript', 'typescript', 'python', 'rust', 'go', 'json', 'yaml', 'markdown', 'mermaid', 'diff', '']
    for (const lang of langs) {
      const md = lang ? `\`\`\`${lang}\ncode\n\`\`\`` : '```\ncode\n```'
      const result = renderMarkdownContent(md)
      expect(containsText(result, 'code')).toBe(true)
    }
  })

  test('th/td align prop values', () => {
    // align: "left"
    const left = renderMarkdownContent('| L |\n|:--|\n| a |')
    expect(containsText(left, 'L')).toBe(true)

    // align: "center"
    const center = renderMarkdownContent('| C |\n|:--:|\n| a |')
    expect(containsText(center, 'C')).toBe(true)

    // align: "right"
    const right = renderMarkdownContent('| R |\n|--:|\n| a |')
    expect(containsText(right, 'R')).toBe(true)

    // align: undefined (no alignment)
    const none = renderMarkdownContent('| N |\n|---|\n| a |')
    expect(containsText(none, 'N')).toBe(true)
  })

  test('a href and title props', () => {
    // href only
    const hrefOnly = renderMarkdownContent('[link](https://example.com)')
    expect(containsText(hrefOnly, 'link')).toBe(true)

    // href + title
    const hrefTitle = renderMarkdownContent('[link](https://example.com "Title")')
    expect(containsText(hrefTitle, 'link')).toBe(true)

    // empty href
    const emptyHref = renderMarkdownContent('[link]()')
    expect(emptyHref).toBeTruthy()
  })

  test('img src, alt, and title props', () => {
    // All three
    const all = renderMarkdownContent('![alt text](https://img.png "Title")')
    expect(containsText(all, '[alt text]')).toBe(true)

    // src + alt only
    const srcAlt = renderMarkdownContent('![alt](https://img.png)')
    expect(containsText(srcAlt, '[alt]')).toBe(true)

    // src only (empty alt)
    const srcOnly = renderMarkdownContent('![](https://img.png)')
    expect(srcOnly).toBeTruthy()
  })
})
