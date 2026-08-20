import * as Command from "@effect/platform/Command"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import {
  DirectoryPathSchema,
  type DirectoryPath,
  type GitInspection,
  type GitRecentFiles,
} from "@magnitudedev/acn-protocol"
import { Chunk, Context, Effect, Layer, Option, Stream } from "effect"

export interface GitInspector {
  readonly inspect: (cwd: DirectoryPath) => Effect.Effect<GitInspection>
  readonly recentFiles: (cwd: DirectoryPath, limit: number) => Effect.Effect<GitRecentFiles>
}

export const GitInspector = Context.GenericTag<GitInspector>("acn/GitInspector")

type GitCommandOutcome =
  | {
      readonly _tag: "completed"
      readonly exitCode: number
      readonly stdout: string
      readonly stderr: string
    }
  | { readonly _tag: "start_failed"; readonly executableMissing: boolean }
  | { readonly _tag: "timed_out" }

const NOT_A_REPOSITORY_MARKER = "fatal: not a git repository"

export const GitInspectorLive: Layer.Layer<
  GitInspector,
  never,
  CommandExecutor.CommandExecutor
> = Layer.effect(
  GitInspector,
  Effect.gen(function* () {
    const executor = yield* CommandExecutor.CommandExecutor

    const collect = <E>(stream: Stream.Stream<Uint8Array, E>): Effect.Effect<string, E> =>
      stream.pipe(
        Stream.decodeText(),
        Stream.runCollect,
        Effect.map((chunk) => Chunk.toReadonlyArray(chunk).join("")),
      )

    const run = (cwd: DirectoryPath, args: ReadonlyArray<string>): Effect.Effect<GitCommandOutcome> =>
      Effect.scoped(
        executor.start(
          // LC_ALL=C keeps git diagnostics stable for the not-a-repository check.
          Command.make("git", "-C", cwd, ...args).pipe(Command.env({ LC_ALL: "C" })),
        ).pipe(
          Effect.flatMap((process) =>
            Effect.all(
              [process.exitCode, collect(process.stdout), collect(process.stderr)],
              { concurrency: 3 },
            ),
          ),
        ),
      ).pipe(
        Effect.map(([exitCode, stdout, stderr]): GitCommandOutcome => ({
          _tag: "completed",
          exitCode: Number(exitCode),
          stdout,
          stderr,
        })),
        Effect.catchTags({
          SystemError: (error) => Effect.succeed<GitCommandOutcome>({
            _tag: "start_failed",
            executableMissing: error.reason === "NotFound",
          }),
          BadArgument: () => Effect.succeed<GitCommandOutcome>({
            _tag: "start_failed",
            executableMissing: false,
          }),
        }),
        Effect.timeoutOption("3 seconds"),
        Effect.map(Option.getOrElse((): GitCommandOutcome => ({ _tag: "timed_out" }))),
      )

    type Classified =
      | { readonly _tag: "output"; readonly stdout: string }
      | { readonly _tag: "git_unavailable" }
      | { readonly _tag: "not_git_repository" }
      | { readonly _tag: "git_inspection_failed" }

    const classify = (outcome: GitCommandOutcome): Classified => {
      switch (outcome._tag) {
        case "start_failed":
          return outcome.executableMissing
            ? { _tag: "git_unavailable" }
            : { _tag: "git_inspection_failed" }
        case "timed_out":
          return { _tag: "git_inspection_failed" }
        case "completed": {
          if (outcome.exitCode === 0) return { _tag: "output", stdout: outcome.stdout }
          return outcome.stderr.startsWith(NOT_A_REPOSITORY_MARKER)
            ? { _tag: "not_git_repository" }
            : { _tag: "git_inspection_failed" }
        }
      }
    }

    return GitInspector.of({
      inspect: Effect.fn("acn.git-inspector.inspect")(function* (cwd) {
        const root = classify(yield* run(cwd, ["rev-parse", "--show-toplevel"]))
        if (root._tag !== "output") return root
        const rootDirectory = root.stdout.trim()
        if (rootDirectory.length === 0) return { _tag: "git_inspection_failed" }

        const head = yield* run(cwd, ["symbolic-ref", "--quiet", "--short", "HEAD"])
        const branch = classify(head)
        if (branch._tag === "git_unavailable") return branch
        if (branch._tag === "output" && branch.stdout.trim().length > 0) {
          return {
            _tag: "git_repository",
            rootDirectory: DirectoryPathSchema.make(rootDirectory),
            head: { _tag: "branch", name: branch.stdout.trim() },
          }
        }
        // symbolic-ref exits 1 without the repository marker on a detached HEAD.
        const revision = classify(yield* run(cwd, ["rev-parse", "--short", "HEAD"]))
        if (revision._tag !== "output" || revision.stdout.trim().length === 0) {
          return revision._tag === "output" ? { _tag: "git_inspection_failed" } : revision
        }
        return {
          _tag: "git_repository",
          rootDirectory: DirectoryPathSchema.make(rootDirectory),
          head: { _tag: "detached", revision: revision.stdout.trim() },
        }
      }),
      recentFiles: Effect.fn("acn.git-inspector.recent-files")(function* (cwd, limit) {
        const outcome = classify(yield* run(cwd, [
          "log",
          "--name-only",
          "--pretty=format:",
          "-n",
          String(limit * 2),
        ]))
        if (outcome._tag !== "output") return outcome
        const seen = new Set<string>()
        const files: string[] = []
        for (const line of outcome.stdout.split(/\r?\n/).map((value) => value.trim())) {
          if (line.length === 0 || seen.has(line)) continue
          seen.add(line)
          files.push(line)
          if (files.length === limit) break
        }
        return { _tag: "recent_git_files", files }
      }),
    })
  }),
)
