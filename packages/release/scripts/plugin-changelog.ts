/** Keep Changesets' new entry aligned with artifact reservation or an unchanged-plugin rollback. */
export const reconcilePluginChangelog = (text: string, from: string, to: string, publish: boolean): string => {
  if (from === to) return text
  const heading = `## ${from}\n`
  const start = text.indexOf(heading)
  if (start === -1) return text
  if (publish) return `${text.slice(0, start)}## ${to}\n${text.slice(start + heading.length)}`
  const next = text.indexOf("\n## ", start + heading.length)
  return text.slice(0, start) + (next === -1 ? "" : text.slice(next + 1))
}
