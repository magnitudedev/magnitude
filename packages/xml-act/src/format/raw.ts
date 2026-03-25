export function rawOpenTag(tagName: string, attrs: ReadonlyMap<string, string>): string {
  const attrString = Array.from(attrs.entries())
    .map(([k, v]) => ` ${k}="${v}"`)
    .join('')
  return `<${tagName}${attrString}>`
}

export function rawCloseTag(tagName: string): string {
  return `</${tagName}>`
}

export function rawSelfCloseTag(tagName: string, attrs: ReadonlyMap<string, string>): string {
  const attrString = Array.from(attrs.entries())
    .map(([k, v]) => ` ${k}="${v}"`)
    .join('')
  return `<${tagName}${attrString} />`
}
