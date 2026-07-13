import { homedir } from 'node:os'
import { resolve, dirname, normalize, join } from 'node:path'
import { exec } from 'node:child_process'
import { promisify } from 'node:util'
import { existsSync } from 'node:fs'

const execAsync = promisify(exec)

/** Total worktree size above which VCS is disabled for non-git directories (10 MB). */
const SIZE_LIMIT_BYTES = 10_000_000

/**
 * Determine whether the shadow VCS should be enabled for the given cwd.
 *
 * Currently disabled — return false. Re-enable by uncommenting the logic below
 * and removing the `return false`.
 */
export async function isVcsAvailable(_cwd: string): Promise<boolean> {
  return false

  // const home = normalize(homedir())
  // const normalizedCwd = normalize(resolve(cwd))
  //
  // // Tier 0: If inside a git repo, always enable
  // if (isInsideGitRepo(normalizedCwd)) return true
  //
  // // Tier 1: Disable if cwd IS home
  // if (normalizedCwd === home) return false
  //
  // // Tier 1: Disable if cwd is above home (e.g. /Users, /, /home)
  // if (home.startsWith(normalizedCwd + '/')) return false
  //
  // // Tier 1: Disable if cwd is a direct child of /
  // const parent = dirname(normalizedCwd)
  // if (parent === '/') return false
  //
  // // Tier 2: Disable if non-git directory is too large
  // try {
  //   const { stdout } = await execAsync(`du -sk "${normalizedCwd}"`)
  //   const kb = parseInt(stdout.trim().split('\t')[0], 10)
  //   if (kb * 1024 > SIZE_LIMIT_BYTES) return false
  // } catch {
  //   // If du fails, we can't determine size — disable to be safe
  //   return false
  // }
  //
  // return true
}

/**
 * Walk up from cwd to find a .git directory.
 * Fast — just checks for directory existence, no git commands.
 */
function isInsideGitRepo(cwd: string): boolean {
  let dir = cwd
  for (;;) {
    if (existsSync(join(dir, '.git'))) return true
    const parent = dirname(dir)
    if (parent === dir) break  // reached root
    dir = parent
  }
  return false
}
