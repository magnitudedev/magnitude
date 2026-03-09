/**
 * Docker Container Management
 *
 * Manages Docker containers for builder-bench scenarios via the Docker CLI.
 * No Docker SDK dependency — all operations use Bun.spawn.
 */

// =============================================================================
// Types
// =============================================================================

export interface DockerContainer {
  containerId: string
  workDir: string
}

export interface ExecResult {
  stdout: string
  stderr: string
  exitCode: number
}

export interface TreeEntry {
  path: string
  name: string
  type: 'file' | 'dir'
  depth: number
}

export interface SearchMatch {
  file: string
  match: string
}

// =============================================================================
// Helpers
// =============================================================================

async function run(args: string[], options?: { stdin?: string; timeout?: number }): Promise<ExecResult> {
  const timeout = options?.timeout ?? 120_000

  const proc = Bun.spawn(args, {
    stdout: 'pipe',
    stderr: 'pipe',
    stdin: options?.stdin !== undefined ? 'pipe' : undefined,
  })

  if (options?.stdin !== undefined && proc.stdin) {
    proc.stdin.write(options.stdin)
    proc.stdin.end()
  }

  const timer = setTimeout(() => proc.kill(), timeout)

  try {
    const [stdout, stderr] = await Promise.all([
      new Response(proc.stdout).text(),
      new Response(proc.stderr).text(),
    ])
    const exitCode = await proc.exited
    return { stdout, stderr, exitCode }
  } finally {
    clearTimeout(timer)
  }
}

// =============================================================================
// Image Management
// =============================================================================

/**
 * Check if a Docker image exists locally.
 */
async function imageExists(tag: string): Promise<boolean> {
  const result = await run(['docker', 'image', 'inspect', tag])
  return result.exitCode === 0
}

/**
 * Build a Docker image from a scenario directory.
 * Skips if the image already exists (use `force` to rebuild).
 */
export async function buildImage(scenarioDir: string, tag: string, force = false): Promise<void> {
  if (!force && await imageExists(tag)) return

  const result = await run(
    ['docker', 'build', '-t', tag, scenarioDir],
    { timeout: 300_000 } // 5 min for builds
  )

  if (result.exitCode !== 0) {
    throw new Error(`Docker build failed for ${tag}:\n${result.stderr}`)
  }
}

// =============================================================================
// Container Lifecycle
// =============================================================================

/**
 * Create and start a new container from an image.
 */
export async function createContainer(imageTag: string, workDir: string): Promise<DockerContainer> {
  const createResult = await run([
    'docker', 'create',
    '--workdir', workDir,
    imageTag,
    'sleep', 'infinity',
  ])

  if (createResult.exitCode !== 0) {
    throw new Error(`Docker create failed: ${createResult.stderr}`)
  }

  const containerId = createResult.stdout.trim()

  const startResult = await run(['docker', 'start', containerId])
  if (startResult.exitCode !== 0) {
    await run(['docker', 'rm', '-f', containerId])
    throw new Error(`Docker start failed: ${startResult.stderr}`)
  }

  return { containerId, workDir }
}

/**
 * Remove and clean up a container.
 */
export async function removeContainer(container: DockerContainer): Promise<void> {
  await run(['docker', 'rm', '-f', container.containerId])
}

// =============================================================================
// Container Operations
// =============================================================================

/**
 * Execute a command inside a running container.
 */
export async function execInContainer(
  container: DockerContainer,
  command: string,
  timeout = 30_000
): Promise<ExecResult> {
  return run(
    ['docker', 'exec', container.containerId, 'sh', '-c', command],
    { timeout }
  )
}

/**
 * Read a file from the container filesystem.
 */
export async function readFile(container: DockerContainer, path: string): Promise<string> {
  const result = await execInContainer(container, `cat ${JSON.stringify(path)}`)
  if (result.exitCode !== 0) {
    throw new Error(`Failed to read ${path}: ${result.stderr}`)
  }
  return result.stdout
}

/**
 * Write a file to the container filesystem.
 */
export async function writeFile(container: DockerContainer, path: string, content: string): Promise<void> {
  // Use docker exec with stdin to handle arbitrary content safely
  const result = await run(
    ['docker', 'exec', '-i', container.containerId, 'sh', '-c', `cat > ${JSON.stringify(path)}`],
    { stdin: content }
  )
  if (result.exitCode !== 0) {
    throw new Error(`Failed to write ${path}: ${result.stderr}`)
  }
}

/**
 * List directory contents in the container.
 */
export async function listDir(
  container: DockerContainer,
  path: string,
  options?: { recursive?: boolean; maxDepth?: number }
): Promise<TreeEntry[]> {
  const recursive = options?.recursive ?? true
  const maxDepth = options?.maxDepth

  let findCmd: string
  if (!recursive) {
    findCmd = `find ${JSON.stringify(path)} -maxdepth 1 -not -path ${JSON.stringify(path)} -printf '%y %P\\n' 2>/dev/null`
  } else if (maxDepth !== undefined) {
    findCmd = `find ${JSON.stringify(path)} -maxdepth ${maxDepth} -not -path ${JSON.stringify(path)} -printf '%y %P\\n' 2>/dev/null`
  } else {
    findCmd = `find ${JSON.stringify(path)} -not -path ${JSON.stringify(path)} -printf '%y %P\\n' 2>/dev/null`
  }

  const result = await execInContainer(container, findCmd)
  if (result.exitCode !== 0 && !result.stdout) {
    return []
  }

  const entries: TreeEntry[] = []
  for (const line of result.stdout.trim().split('\n')) {
    if (!line) continue
    const typeChar = line[0]
    const relPath = line.slice(2)
    if (!relPath) continue

    const parts = relPath.split('/')
    entries.push({
      path: relPath,
      name: parts[parts.length - 1],
      type: typeChar === 'd' ? 'dir' : 'file',
      depth: parts.length - 1,
    })
  }

  return entries
}

/**
 * Search file contents in the container using grep.
 */
export async function searchFiles(
  container: DockerContainer,
  pattern: string,
  searchPath = '.',
  glob?: string,
): Promise<SearchMatch[]> {
  let cmd: string
  if (glob) {
    cmd = `grep -rn ${JSON.stringify(pattern)} --include=${JSON.stringify(glob)} ${JSON.stringify(searchPath)} 2>/dev/null`
  } else {
    cmd = `grep -rn ${JSON.stringify(pattern)} ${JSON.stringify(searchPath)} 2>/dev/null`
  }

  const result = await execInContainer(container, cmd)
  // grep returns exit 1 when no matches — not an error
  if (result.exitCode > 1) {
    return []
  }

  const matches: SearchMatch[] = []
  for (const line of result.stdout.trim().split('\n')) {
    if (!line) continue
    const colonIdx = line.indexOf(':')
    if (colonIdx === -1) continue
    const file = line.slice(0, colonIdx)
    const rest = line.slice(colonIdx + 1)
    matches.push({ file, match: rest })
  }

  return matches
}

/**
 * Hash files in the container for integrity checking.
 * Returns a map of path → sha256 hash.
 */
export async function hashFiles(
  container: DockerContainer,
  paths: string[],
): Promise<Record<string, string>> {
  if (paths.length === 0) return {}

  const cmd = `sha256sum ${paths.map(p => JSON.stringify(p)).join(' ')} 2>/dev/null`
  const result = await execInContainer(container, cmd)

  const hashes: Record<string, string> = {}
  for (const line of result.stdout.trim().split('\n')) {
    if (!line) continue
    const parts = line.split(/\s+/)
    if (parts.length >= 2) {
      hashes[parts[1]] = parts[0]
    }
  }

  return hashes
}

/**
 * Snapshot all source files in the container workspace.
 * Returns a map of relative path → file content.
 * Skips node_modules, __pycache__, .git, bun.lockb.
 */
export async function snapshotFiles(container: DockerContainer): Promise<Record<string, string>> {
  const listResult = await execInContainer(
    container,
    `find . -type f -not -path '*/node_modules/*' -not -path '*/__pycache__/*' -not -path '*/.git/*' -not -name 'bun.lockb'`
  )
  const files = listResult.stdout.trim().split('\n').filter(Boolean)
  const snapshot: Record<string, string> = {}

  for (const file of files) {
    const result = await execInContainer(container, `cat ${JSON.stringify(file)} 2>/dev/null`)
    if (result.exitCode === 0) {
      snapshot[file] = result.stdout
    }
  }

  return snapshot
}

/**
 * Diff two snapshots. Returns a human-readable unified-style diff string.
 */
export function diffSnapshots(
  before: Record<string, string>,
  after: Record<string, string>,
): string {
  const diffs: string[] = []

  // Modified and deleted files
  for (const file of Object.keys(before)) {
    if (!(file in after)) {
      diffs.push(`--- a/${file}\n+++ /dev/null (deleted)`)
      for (const line of before[file].split('\n')) {
        diffs.push(`-${line}`)
      }
      diffs.push('')
    } else if (before[file] !== after[file]) {
      diffs.push(`--- a/${file}\n+++ b/${file}`)
      const origLines = before[file].split('\n')
      const currLines = after[file].split('\n')
      const maxLen = Math.max(origLines.length, currLines.length)
      for (let i = 0; i < maxLen; i++) {
        const orig = origLines[i]
        const curr = currLines[i]
        if (orig === curr) continue
        if (orig !== undefined && curr !== undefined) {
          diffs.push(`-${orig}`)
          diffs.push(`+${curr}`)
        } else if (orig !== undefined) {
          diffs.push(`-${orig}`)
        } else {
          diffs.push(`+${curr}`)
        }
      }
      diffs.push('')
    }
  }

  // New files
  for (const file of Object.keys(after)) {
    if (!(file in before)) {
      diffs.push(`--- /dev/null\n+++ b/${file} (new)`)
      for (const line of after[file].split('\n')) {
        diffs.push(`+${line}`)
      }
      diffs.push('')
    }
  }

  return diffs.join('\n')
}

// =============================================================================
// Pre-flight Check
// =============================================================================

/**
 * Verify Docker is available and running.
 */
export async function checkDocker(): Promise<boolean> {
  const result = await run(['docker', 'info'])
  return result.exitCode === 0
}
