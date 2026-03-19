export function expandWorkspacePath(path: string, workspacePath: string): string {
  if (path === '$M' || path === '${M}') return workspacePath
  if (path.startsWith('$M/')) return workspacePath + path.slice(2)
  if (path.startsWith('${M}/')) return workspacePath + path.slice(4)

  return path
}
