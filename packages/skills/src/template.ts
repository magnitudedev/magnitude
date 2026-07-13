export function resolveTemplates(text: string, submissions: ReadonlyMap<string, string>): string {
  return text.replace(/{{\s*([^}]+?)\s*}}/g, (match, key: string) => submissions.get(key) ?? match)
}
