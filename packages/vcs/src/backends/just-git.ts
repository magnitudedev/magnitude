import { Effect } from "effect"
import * as path from "node:path"

import type { VcsBackend } from "../backend"
import type { Delta, RestoreScope } from "../types"
import { VcsBackendError } from "../errors"
import { createRestoreScopePredicate } from "../path-selector"
import { createContentPatch } from "../diff-content"

// ── just-git SDK + exec ─────────────────────────────────────────────
import { type FileSystem, findRepo, type GitContext, createGit } from "just-git"
import {
  createTreeAccessor,
  diffTrees as diffTreesRepo,
  flattenTree as flattenTreeRepo,
  readBlob as readBlobRepo,
  readCommit as readCommitRepo,
  readHead as readHeadRepo,
  walkCommitHistory,
} from "just-git/repo"

// ── Real-disk FileSystem adapter (module-level, for production) ──────

import {
  readFile as fsReadFile,
  stat as fsStat,
  lstat as fsLstat,
  readdir as fsReaddir,
  mkdir as fsMkdir,
  rm as fsRm,
  writeFile as nodeWriteFile,
} from "node:fs/promises"
import { existsSync } from "node:fs"

type FlatTreeEntry = Awaited<ReturnType<typeof flattenTreeRepo>>[number]

export const realFs: FileSystem = {
  async readFile(p) {
    const buf = await fsReadFile(p)
    return new TextDecoder().decode(buf)
  },
  async readFileBuffer(p) {
    return new Uint8Array(await fsReadFile(p))
  },
  async writeFile(p, content) {
    await nodeWriteFile(p, content)
  },
  async exists(p) {
    return existsSync(p)
  },
  async stat(p) {
    const s = await fsStat(p)
    return {
      isFile: s.isFile(),
      isDirectory: s.isDirectory(),
      isSymbolicLink: s.isSymbolicLink(),
      mode: s.mode,
      size: s.size,
      mtime: s.mtime,
    }
  },
  async lstat(p) {
    const s = await fsLstat(p)
    return {
      isFile: s.isFile(),
      isDirectory: s.isDirectory(),
      isSymbolicLink: s.isSymbolicLink(),
      mode: s.mode,
      size: s.size,
      mtime: s.mtime,
    }
  },
  async readlink(p) {
    const { readlink } = await import("node:fs/promises")
    return readlink(p)
  },
  async symlink(target, p) {
    const { symlink } = await import("node:fs/promises")
    await symlink(target, p)
  },
  async mkdir(p, options) {
    await fsMkdir(p, { recursive: options?.recursive ?? false })
  },
  async readdir(p) {
    const entries = await fsReaddir(p, { withFileTypes: true })
    return entries.map((e) => e.name)
  },
  async rm(p, options) {
    await fsRm(p, {
      recursive: options?.recursive ?? false,
      force: options?.force ?? false,
    })
  },
}

// ── Helpers ──────────────────────────────────────────────────────────

function assertCtx(ctx: GitContext | undefined): asserts ctx is GitContext {
  if (!ctx) throw new Error("ShadowVcs backend not initialised")
}

function vcsErr(operation: string, cause: unknown): VcsBackendError {
  const message = cause instanceof Error ? cause.message : String(cause)
  return new VcsBackendError({ operation, message, cause })
}

function eff<A>(
  operation: string,
  fn: () => Promise<A>,
): Effect.Effect<A, VcsBackendError> {
  return Effect.tryPromise({
    try: fn,
    catch: (cause) => vcsErr(operation, cause),
  })
}

// ── Backend factory ─────────────────────────────────────────────────

/**
 * Create a VcsBackend backed by just-git.
 *
 * @param worktreePath - Path to the project directory being tracked
 * @param gitDirPath - Path where the shadow .git directory should live
 * @param fs - FileSystem to use for worktree operations. Defaults to realFs.
 *             Pass a MemoryFileSystem for in-memory testing.
 */
/** Default git commit author for the shadow VCS. */
const DEFAULT_AUTHOR = { name: "Magnitude Agent", email: "agent@magnitude.dev" } as const

export async function createJustGitBackend(
  worktreePath: string,
  gitDirPath: string,
  fs: FileSystem,
  options?: { readonly defaultAuthor?: { readonly name: string; readonly email: string } },
): Promise<VcsBackend> {
  const author = options?.defaultAuthor ?? DEFAULT_AUTHOR
  // ── Init ──
  // just-git's `git init` ignores gitDir and creates .git in cwd.
  // We must create .git in the worktree, then move it to gitDirPath.
  await fs.mkdir(`${gitDirPath}/objects`, { recursive: true })
  await fs.mkdir(`${gitDirPath}/refs/heads`, { recursive: true })
  await fs.mkdir(`${gitDirPath}/refs/tags`, { recursive: true })
  await fs.mkdir(`${gitDirPath}/info`, { recursive: true })

  await fs.writeFile(`${gitDirPath}/HEAD`, "ref: refs/heads/main\n")

  const config = `[core]
\trepositoryformatversion = 0
\tfilemode = false
\tbare = false
\tlogallrefupdates = true
\tworktree = ${worktreePath}
[user]
\tname = ${author.name}
\temail = ${author.email}
`
  await fs.writeFile(`${gitDirPath}/config`, config)

  await fs.writeFile(
    `${gitDirPath}/description`,
    "Unnamed repository; shadow VCS for Magnitude agent\n",
  )

  // Exclude the storage directory so git add/status ignore it.
  const storageRelPath = path.relative(worktreePath, gitDirPath).replace(/\/\.git$/, "")
  await fs.writeFile(
    `${gitDirPath}/info/exclude`,
    `# Magnitude shadow VCS excludes\n${storageRelPath}/\n.git\n`,
  )

  const found = await findRepo(fs, gitDirPath)
  if (!found) {
    throw new Error("Failed to resolve shadow git context after init")
  }

  const ctx: GitContext = { ...found, workTree: worktreePath }

  // Git exec instance — uses objectStore + refStore + gitDir to bypass
  // filesystem .git discovery. This means exec commands always target
  // our shadow repo even when a user's .git exists in the worktree.
  // info/exclude is properly read from our shadow .git.
  const identity = author

  const git = createGit({
    fs,
    cwd: worktreePath,
    gitDir: gitDirPath,
    objectStore: found.objectStore,
    refStore: found.refStore,
    identity,
  })
  const workFs = fs

  // Helper for exec commands
  async function execGit(command: string): Promise<{ stdout: string; stderr: string; exitCode: number }> {
    const result = await git.exec(command, { fs, cwd: worktreePath })
    return result
  }

  // Helper for status parsing
  function parsePorcelainLine(line: string): { xy: string; path: string } | null {
    if (line.length < 3) return null
    const x = line.charAt(0)
    const y = line.charAt(1)
    const sep = line.charAt(2)
    const rawPath = sep === " "
      ? line.substring(3).trimEnd()
      : line.substring(2).trimEnd()
    const currentPath = rawPath.includes(" -> ")
      ? rawPath.slice(rawPath.lastIndexOf(" -> ") + 4)
      : rawPath

    return { xy: x + y, path: unquoteGitPath(currentPath) }
  }

  function unquoteGitPath(filePath: string): string {
    if (!filePath.startsWith('"') || !filePath.endsWith('"')) return filePath

    let result = ""
    const body = filePath.slice(1, -1)
    for (let i = 0; i < body.length; i++) {
      const char = body[i]!
      if (char !== "\\") {
        result += char
        continue
      }

      const next = body[++i]
      if (next === undefined) result += "\\"
      else if (next === "n") result += "\n"
      else if (next === "t") result += "\t"
      else result += next
    }
    return result
  }

  function isInternalPath(filePath: string): boolean {
    const normalized = filePath.replace(/\/+$/, "")
    return normalized.startsWith(storageRelPath + "/")
      || normalized === storageRelPath
      || normalized.split("/").includes(".git")
  }

  function fullWorktreePath(relativePath: string): string {
    const root = path.resolve(worktreePath)
    const fullPath = path.resolve(worktreePath, relativePath)
    if (fullPath !== root && !fullPath.startsWith(root + path.sep)) {
      throw new Error(`refusing to restore path outside worktree: ${relativePath}`)
    }
    return fullPath
  }

  async function writeTreeEntry(entry: FlatTreeEntry): Promise<void> {
    if (isInternalPath(entry.path)) return

    const fullPath = fullWorktreePath(entry.path)
    await workFs.mkdir(path.dirname(fullPath), { recursive: true })
    const content = await readBlobRepo(ctx, entry.hash)

    if (entry.mode === "120000") {
      const target = new TextDecoder().decode(content)
      await workFs.rm(fullPath, { recursive: true, force: true }).catch(() => {})
      if (workFs.symlink) {
        await workFs.symlink(target, fullPath)
      } else {
        await workFs.writeFile(fullPath, target)
      }
      return
    }

    if (workFs.lstat) {
      try {
        const stat = await workFs.lstat(fullPath)
        if (stat.isDirectory || stat.isSymbolicLink) {
          await workFs.rm(fullPath, { recursive: true, force: true })
        }
      } catch {
        // Missing path is fine.
      }
    }
    await workFs.writeFile(fullPath, content)
  }

  async function getHeadTreeEntries(): Promise<FlatTreeEntry[]> {
    const head = await readHeadRepo(ctx)
    if (!head.hash) return []
    const commit = await readCommitRepo(ctx, head.hash)
    return await flattenTreeRepo(ctx, commit.tree)
  }

  async function expandWorktreeFilePaths(filePath: string): Promise<string[]> {
    const normalized = filePath.replace(/\/+$/, "")
    if (!normalized || isInternalPath(normalized)) return []

    const fullPath = fullWorktreePath(normalized)
    let stat
    try {
      stat = workFs.lstat ? await workFs.lstat(fullPath) : await workFs.stat(fullPath)
    } catch {
      return [normalized]
    }

    if (stat.isFile || stat.isSymbolicLink) return [normalized]
    if (!stat.isDirectory) return []

    const children = await workFs.readdir(fullPath)
    const files = await Promise.all(
      children.map((child) => expandWorktreeFilePaths(`${normalized}/${child}`)),
    )
    return files.flat().sort()
  }

  return {
    buildCommit: (options) =>
      eff("buildCommit", async () => {
        assertCtx(ctx)
        const addResult = await execGit("add .")
        if (addResult.exitCode !== 0) {
          throw new Error(`git add failed: ${addResult.stderr}`)
        }

        const commitResult = await execGit(`commit -m "${options.message}" --allow-empty`)
        if (commitResult.exitCode !== 0) {
          throw new Error(`git commit failed: ${commitResult.stderr}`)
        }

        const head = await readHeadRepo(ctx)
        if (head.hash) return head.hash
        throw new Error("Could not read HEAD after commit")
      }),

    getChangedFiles: () =>
      eff("getChangedFiles", async () => {
        assertCtx(ctx)
        const result = await execGit("status --porcelain")
        if (result.exitCode !== 0) {
          throw new Error(`git status failed: ${result.stderr}`)
        }

        const diffs: Array<{ path: string; status: "added" | "deleted" | "modified" }> = []
        const lines = result.stdout.trim().split("\n").filter(Boolean)

        for (const line of lines) {
          const parsed = parsePorcelainLine(line)
          if (!parsed) continue
          const { xy, path: filePath } = parsed
          const expandedPaths = await expandWorktreeFilePaths(filePath)
          const paths = expandedPaths.length > 0 ? expandedPaths : [filePath.replace(/\/+$/, "")]

          // X = index status, Y = worktree status
          const x = xy[0]!
          const y = xy[1]!

          // Determine overall status
          let status: "added" | "deleted" | "modified"
          if (x === "A" || y === "A" || x === "?" || y === "?") {
            status = "added"
          } else if (x === "D" || y === "D") {
            status = "deleted"
          } else if (x === "M" || y === "M" || x === "R" || y === "R" || x === "C" || y === "C") {
            status = "modified"
          } else {
            // Unknown status, skip
            continue
          }

          for (const changedPath of paths) {
            if (changedPath && !isInternalPath(changedPath)) {
              diffs.push({ path: changedPath, status })
            }
          }
        }

        return diffs
      }),

    diffTree: (fromTreeHash, toTreeHash) =>
      eff("diffTree", async () => {
        assertCtx(ctx)
        const entries = await diffTreesRepo(ctx, fromTreeHash, toTreeHash)

        // Batch all blob reads — collect unique hashes, read once, reuse
        const hashSet = new Set<string>()
        for (const e of entries) {
          if (e.oldHash) hashSet.add(e.oldHash)
          if (e.newHash) hashSet.add(e.newHash)
        }
        const blobCache = new Map<string, Uint8Array>()
        await Promise.all(
          Array.from(hashSet, async (hash) => {
            blobCache.set(hash, await readBlobRepo(ctx, hash))
          }),
        )

        const files: Array<{
          path: string
          status: "added" | "deleted" | "modified" | "renamed"
          oldPath?: string
          diff: string
        }> = []
        let additions = 0
        let deletions = 0
        let modifications = 0
        let renames = 0

        for (const e of entries) {
          if (e.status === "added") {
            additions++
            const newContent = e.newHash ? blobCache.get(e.newHash)! : null
            const diff = createContentPatch(e.path, null, newContent)
            files.push({ path: e.path, status: "added", diff })
          } else if (e.status === "deleted") {
            deletions++
            const oldContent = e.oldHash ? blobCache.get(e.oldHash)! : null
            const diff = createContentPatch(e.path, oldContent, null)
            files.push({ path: e.path, status: "deleted", diff })
          } else if (e.status === "modified") {
            modifications++
            const oldContent = e.oldHash ? blobCache.get(e.oldHash)! : null
            const newContent = e.newHash ? blobCache.get(e.newHash)! : null
            const diff = createContentPatch(e.path, oldContent, newContent)
            files.push({ path: e.path, status: "modified", diff })
          }
        }

        return { additions, deletions, modifications, renames, files }
      }),

    extractTree: (treeHashOrCommitHash, scope) =>
      eff("extractTree", async () => {
        assertCtx(ctx)
        const predicate = createRestoreScopePredicate(scope)

        // Resolve commit hash to tree hash if needed
        let treeHash = treeHashOrCommitHash
        try {
          const commitObj = await readCommitRepo(ctx, treeHashOrCommitHash)
          treeHash = commitObj.tree
        } catch {
          // If readCommit fails, assume it's already a tree hash
        }

        if (!scope || scope.kind === "full") {
          // Full restore — all index-based, no filesystem walking:
          // 1. git rm -rf . — removes all tracked files from worktree + index
          // 2. Materialize target tree to worktree via SDK (object store only)
          // 3. git add . — stages the new files into index
          // 4. git clean -fd — removes any truly untracked leftovers
          const rmResult = await execGit("rm -rf .")
          // "pathspec '.' did not match any files" is harmless (empty index)
          if (rmResult.exitCode !== 0 && !rmResult.stderr.includes("did not match any files")) {
            throw new Error(`git rm failed: ${rmResult.stderr}`)
          }

          // Materialize target tree to worktree
          const flat = (await flattenTreeRepo(ctx, treeHash))
            .filter((entry) => !isInternalPath(entry.path))
          for (const entry of flat) {
            await writeTreeEntry(entry)
          }

          // Stage new files + clean untracked leftovers
          await execGit("add .")
          const cleanResult = await execGit("clean -fd")
          if (cleanResult.exitCode !== 0) {
            throw new Error(`git clean failed: ${cleanResult.stderr}`)
          }

          return flat.map((e) => e.path).filter(predicate)
        }

        // Scoped restore: write specific files from tree, delete matching tracked
        // files that do not exist in the target tree.
        const flat = (await flattenTreeRepo(ctx, treeHash))
          .filter((entry) => !isInternalPath(entry.path))
        const matched = flat.filter((e) => predicate(e.path))

        const restored: string[] = []
        for (const entry of matched) {
          await writeTreeEntry(entry)
          restored.push(entry.path)
        }

        // Delete files in scope that are tracked but NOT in the target tree.
        // Use the current HEAD tree, not `git ls-files` text output, so quoted
        // unicode/space paths and symlinks are handled structurally.
        const treePaths = new Set(matched.map((e) => e.path))
        const currentEntries = await getHeadTreeEntries()
        for (const trackedPath of currentEntries.map((entry) => entry.path)) {
          if (!isInternalPath(trackedPath) && predicate(trackedPath) && !treePaths.has(trackedPath)) {
            await workFs.rm(fullWorktreePath(trackedPath), { recursive: true, force: true }).catch(() => {})
          }
        }

        // Sync index to new worktree state
        await execGit("add .")

        return restored
      }),

    readFileAt: (treeHash, filePath) =>
      eff("readFileAt", async () => {
        assertCtx(ctx)
        const accessor = createTreeAccessor(ctx, treeHash)
        return await accessor.readFileBytes(filePath)
      }),

    walkTree: (treeHash) =>
      eff("walkTree", async () => {
        assertCtx(ctx)
        const flat = await flattenTreeRepo(ctx, treeHash)
        return flat.map((e) => ({ path: e.path, hash: e.hash, mode: e.mode }))
      }),

    readRef: (ref) =>
      eff("readRef", async () => {
        assertCtx(ctx)
        const result = await ctx.refStore.readRef(ref)
        if (!result) return null
        if (result.type === "direct") return result.hash
        if (result.type === "symbolic") {
          const resolved = await ctx.refStore.readRef(result.target)
          if (resolved && resolved.type === "direct") return resolved.hash
        }
        return null
      }),

    updateRef: (ref, commitHash) =>
      eff("updateRef", async () => {
        assertCtx(ctx)
        await ctx.refStore.writeRef(ref, { type: "direct", hash: commitHash })
      }),

    deleteRef: (ref) =>
      eff("deleteRef", async () => {
        assertCtx(ctx)
        await ctx.refStore.deleteRef(ref)
      }),

    listRefs: (prefix) =>
      eff("listRefs", async () => {
        assertCtx(ctx)
        const entries = await ctx.refStore.listRefs(prefix)
        return entries.map((e) => ({ ref: e.name, hash: e.hash }))
      }),

    readHead: () =>
      eff("readHead", async () => {
        assertCtx(ctx)
        const info = await readHeadRepo(ctx)
        if (info.hash === null) {
          if (info.ref === null && info.branch === null) {
            return { kind: "unborn" as const }
          }
          return { kind: "unborn" as const }
        }
        if (info.ref === null && info.branch === null) {
          return { kind: "direct" as const, hash: info.hash }
        }
        return { kind: "symbolic" as const, target: info.ref ?? "refs/heads/main" }
      }),

    walkHistory: (options) =>
      eff("walkHistory", async () => {
        assertCtx(ctx)
        const headInfo = await readHeadRepo(ctx)
        const start = options?.start ?? headInfo.hash ?? "HEAD"
        const limit = options?.limit
        const pathFilter = options?.pathFilter

        const commits: Array<{
          hash: string
          message: string
          tree: string
          parents: string[]
          author: { name: string; email: string; timestamp: number; timezone: string }
          committer: { name: string; email: string; timestamp: number; timezone: string }
        }> = []

        let count = 0
        for await (const c of walkCommitHistory(ctx, start, {
          firstParent: true,
          paths: pathFilter ? [pathFilter] : undefined,
          limit,
        })) {
          commits.push({
            hash: c.hash,
            message: c.message,
            tree: c.tree,
            parents: c.parents,
            author: c.author,
            committer: c.committer,
          })
          if (limit !== undefined && ++count >= limit) break
        }

        return commits
      }),

    readWorktreeFile: (relativePath) =>
      eff("readWorktreeFile", async () => {
        assertCtx(ctx)
        const fullPath = fullWorktreePath(relativePath)
        try {
          if (workFs.lstat && workFs.readlink) {
            const stat = await workFs.lstat(fullPath)
            if (stat.isSymbolicLink) {
              return new TextEncoder().encode(await workFs.readlink(fullPath))
            }
          }
          return await workFs.readFileBuffer(fullPath)
        } catch {
          return null
        }
      }),

    dispose: () => Effect.void,
  }
}
