import { Context, Layer } from "effect"
import type { FileSystem } from "just-git"
import { realFs } from "./backends/just-git"

/**
 * FileSystem dependency for the VCS package.
 *
 * Follows the codebase's standard Effect DI pattern (see packages/agent/src/services/fs.ts):
 * - Context.Tag defines the service
 * - Production: VcsFsLive provides the real filesystem adapter
 * - Tests: Layer.succeed(VcsFs, new MemoryFileSystem()) provides in-memory FS
 *
 * The FileSystem is resolved at Layer construction time, not per-method.
 * The backend captures it as a closure variable (_workFs).
 */
export class VcsFs extends Context.Tag("VcsFs")<VcsFs, FileSystem>() {}

/** Production layer: provides the real node:fs adapter. */
export const VcsFsLive = Layer.succeed(VcsFs, realFs)
