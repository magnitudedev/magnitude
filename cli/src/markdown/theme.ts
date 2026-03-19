import { blue, slate, green, violet } from '../utils/palette'

export interface SyntaxColors {
  keyword: string
  string: string
  number: string
  comment: string
  function: string
  variable: string
  type: string
  operator: string
  property: string
  punctuation: string
  literal: string
  default: string
}

export interface MarkdownPalette {
  inlineCodeFg: string
  codeBackground: string
  codeBorderColor: string
  codeHeaderFg: string
  headingFg: Record<number, string>
  listBulletFg: string
  blockquoteBorderFg: string
  blockquoteTextFg: string
  dividerFg: string
  codeTextFg: string
  codeMonochrome: boolean
  linkFg: string
  syntax: SyntaxColors
}

export interface CharacterHighlightRange {
  start: number
  end: number
  backgroundColor: string
}

export interface MarkdownRenderOptions {
  palette?: Partial<MarkdownPalette>
  codeBlockWidth?: number
  highlightRanges?: CharacterHighlightRange[]
}

const defaultSyntaxColors: SyntaxColors = {
  keyword: violet[300],
  string: green[300],
  number: blue[300],
  comment: slate[500],
  function: blue[400],
  variable: slate[200],
  type: green[300],
  operator: slate[400],
  property: slate[200],
  punctuation: slate[500],
  literal: blue[300],
  default: slate[100],
}

const defaultPalette: MarkdownPalette = {
  inlineCodeFg: green[300],
  codeBackground: 'transparent',
  codeBorderColor: slate[400],
  codeHeaderFg: slate[500],
  headingFg: {
    1: blue[400],
    2: blue[400],
    3: blue[400],
    4: blue[400],
    5: blue[400],
    6: blue[400],
  },
  listBulletFg: slate[400],
  blockquoteBorderFg: slate[700],
  blockquoteTextFg: slate[200],
  dividerFg: slate[800],
  codeTextFg: slate[100],
  codeMonochrome: false,
  linkFg: blue[400],
  syntax: defaultSyntaxColors,
}

export const buildMergedPalette = (overrides?: Partial<MarkdownPalette>): MarkdownPalette => {
  const palette: MarkdownPalette = {
    ...defaultPalette,
    headingFg: { ...defaultPalette.headingFg },
    syntax: { ...defaultPalette.syntax },
  }

  if (!overrides) {
    return palette
  }

  const { headingFg, syntax, ...rest } = overrides
  Object.assign(palette, rest)

  if (headingFg) {
    palette.headingFg = {
      ...palette.headingFg,
      ...headingFg,
    }
  }

  if (syntax) {
    palette.syntax = {
      ...palette.syntax,
      ...syntax,
    }
  }

  return palette
}
