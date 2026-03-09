import { describe, test, expect } from 'bun:test'
import { validateAndApply, toEditDiff } from '../../util/edit'

describe('string find-replace edit', () => {
  const sampleContent = [
    'import { foo } from "foo";',
    'import { bar } from "bar";',
    '',
    'function main() {',
    '  const x = 1;',
    '  const y = 2;',
    '  const z = 3;',
    '  return x + y + z;',
    '}',
    '',
    'export default main;',
  ].join('\n')

  // ===========================================================================
  // Single-line replace
  // ===========================================================================

  describe('single-line replace', () => {
    test('replaces one line', () => {
      const applied = validateAndApply(sampleContent, '  const x = 1;', '  const x = 10;', false)
      expect(applied.removedLines).toEqual(['  const x = 1;'])
      expect(applied.addedLines).toEqual(['  const x = 10;'])
      expect(applied.replaceCount).toBe(1)
      expect(applied.result).toContain('  const x = 10;')
      expect(applied.result).not.toContain('  const x = 1;')
    })
  })

  // ===========================================================================
  // Multi-line replace
  // ===========================================================================

  describe('multi-line replace', () => {
    test('replaces multiple lines', () => {
      const applied = validateAndApply(
        sampleContent,
        '  const x = 1;\n  const y = 2;\n  const z = 3;',
        '  const x = 10;\n  const y = 20;\n  const z = 30;',
        false,
      )
      expect(applied.removedLines).toEqual(['  const x = 1;', '  const y = 2;', '  const z = 3;'])
      expect(applied.addedLines).toEqual(['  const x = 10;', '  const y = 20;', '  const z = 30;'])
      expect(applied.result).toContain('  const x = 10;')
    })

    test('shrinking replace (3 lines to 1)', () => {
      const applied = validateAndApply(
        sampleContent,
        '  const x = 1;\n  const y = 2;\n  const z = 3;',
        '  const sum = 6;',
        false,
      )
      expect(applied.result).toContain('  const sum = 6;')
      expect(applied.result.split('\n').length).toBe(9) // 11 - 3 + 1
    })

    test('expanding replace (1 line to 3)', () => {
      const applied = validateAndApply(
        sampleContent,
        '  return x + y + z;',
        '  const sum = x + y + z;\n  console.log(sum);\n  return sum;',
        false,
      )
      expect(applied.result).toContain('  console.log(sum);')
      expect(applied.result.split('\n').length).toBe(13) // 11 - 1 + 3
    })
  })

  // ===========================================================================
  // Delete
  // ===========================================================================

  describe('delete', () => {
    test('deletes text with empty newString', () => {
      const applied = validateAndApply(
        sampleContent,
        'import { bar } from "bar";\n',
        '',
        false,
      )
      expect(applied.removedLines).toEqual(['import { bar } from "bar";', ''])
      expect(applied.addedLines).toEqual([''])
      expect(applied.result).not.toContain('import { bar }')
    })
  })

  // ===========================================================================
  // Replace all
  // ===========================================================================

  describe('replaceAll', () => {
    test('replaces all occurrences', () => {
      const content = 'const a = foo;\nconst b = foo;\nconst c = foo;'
      const applied = validateAndApply(content, 'foo', 'bar', true)
      expect(applied.replaceCount).toBe(3)
      expect(applied.result).toBe('const a = bar;\nconst b = bar;\nconst c = bar;')
    })

    test('replaces single occurrence with replaceAll flag', () => {
      const applied = validateAndApply(sampleContent, '  const x = 1;', '  const x = 10;', true)
      expect(applied.replaceCount).toBe(1)
      expect(applied.result).toContain('  const x = 10;')
    })
  })

  // ===========================================================================
  // Multiple sequential edits
  // ===========================================================================

  describe('multiple sequential edits', () => {
    test('second edit sees first edit result', () => {
      const first = validateAndApply(sampleContent, '  const x = 1;', '  const x = 10;', false)
      const second = validateAndApply(first.result, '  const y = 2;', '  const y = 20;', false)

      expect(second.result).toContain('  const x = 10;')
      expect(second.result).toContain('  const y = 20;')
    })
  })

  // ===========================================================================
  // Validation errors
  // ===========================================================================

  describe('validation errors', () => {
    test('throws when old string not found', () => {
      expect(() =>
        validateAndApply(sampleContent, 'this does not exist in the file', 'replacement', false),
      ).toThrow('not found')
    })

    test('throws when old string matches multiple times without replaceAll', () => {
      const content = 'const a = 1;\nconst b = 1;\nconst c = 1;'
      expect(() =>
        validateAndApply(content, 'const', 'let', false),
      ).toThrow('3 locations')
    })
  })

  // ===========================================================================
  // Diff output
  // ===========================================================================

  describe('toEditDiff', () => {
    test('converts applied edit to diff', () => {
      const applied = validateAndApply(sampleContent, '  const x = 1;', '  const x = 10;', false)
      const diff = toEditDiff(applied)
      expect(diff.startLine).toBe(5)
      expect(diff.removedLines).toEqual(['  const x = 1;'])
      expect(diff.addedLines).toEqual(['  const x = 10;'])
    })

    test('startLine is correct for later lines', () => {
      const applied = validateAndApply(sampleContent, '  return x + y + z;', '  return 0;', false)
      const diff = toEditDiff(applied)
      expect(diff.startLine).toBe(8)
    })
  })
})
