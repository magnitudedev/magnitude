export async function runGitCommand(args: string[], cwd: string, timeoutMs = 5000): Promise<string | null> {
  const proc = Bun.spawn(['git', ...args], { cwd, stdout: 'pipe', stderr: 'ignore' })
  const stdoutPromise = new Response(proc.stdout).text()
  const timeout = setTimeout(() => proc.kill(), timeoutMs)
  const exit = await proc.exited
  clearTimeout(timeout)
  if (exit !== 0) return null
  return (await stdoutPromise).trim()
}