import { describe, expect, test } from 'bun:test'
import { normalizeModelOutput, normalizeQuotesInString } from './output-normalization'

describe('normalizeQuotesInString', () => {
  test('converts curly single and double quotes', () => {
    expect(normalizeQuotesInString('\u201chello\u201d it\u2019s fine')).toBe(`"hello" it's fine`)
  })

  test('returns unchanged string when no curly quotes', () => {
    expect(normalizeQuotesInString('plain ascii')).toBe('plain ascii')
  })
})

describe('normalizeModelOutput', () => {
  test('normalizes nested object/array strings', () => {
    const input = {
      title: '\u201cRoadmap\u201d',
      nested: [{ text: 'it\u2019s done' }, '\u201cquoted\u201d'],
    }
    const out = normalizeModelOutput(input)
    expect(out).toEqual({
      title: '"Roadmap"',
      nested: [{ text: "it's done" }, '"quoted"'],
    })
  })

  test('leaves non-plain objects unchanged', () => {
    const d = new Date('2020-01-01')
    const out = normalizeModelOutput({ d })
    expect(out.d).toBe(d)
  })

  test('handles cycles safely', () => {
    const a: any = { text: '\u201cx\u201d' }
    a.self = a
    const out: any = normalizeModelOutput(a)
    expect(out.text).toBe('"x"')
    expect(out.self).toBe(out)
  })
})

describe('normalizeModelOutput - array cycles', () => {
  test('handles array cycles safely', () => {
    const a: any[] = ['\u201cx\u201d']
    a.push(a)
    const out: any = normalizeModelOutput(a)
    expect(out[0]).toBe('"x"')
    expect(out[1]).toBe(out)
  })
})