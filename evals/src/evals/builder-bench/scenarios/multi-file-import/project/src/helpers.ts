export function capitalize(str: string): string {
  if (str.length === 0) return str
  return str[0].toLowerCase() + str.slice(1)
}

export function slugify(str: string): string {
  return str.toLowerCase().replace(/\s+/g, '-').replace(/[^a-z0-9-]/g, '')
}
