import { Effect, Stream, Chunk } from "effect"
import * as Command from "@effect/platform/Command"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as FileSystem from "@effect/platform/FileSystem"
import { loadSkills } from "@magnitudedev/skills"
import type {
  RunBashResult,
  SessionError,
  SkillContent,
  SkillListEntry,
} from "@magnitudedev/acn-protocol"
import { SessionOperationFailed } from "@magnitudedev/acn-protocol"

export function listSkills(
  cwd: string
): Effect.Effect<Array<SkillListEntry>, SessionError, FileSystem.FileSystem> {
  return Effect.gen(function* () {
    const skills = yield* Effect.tryPromise({
      try: () => loadSkills(cwd),
      catch: (cause) =>
        new SessionOperationFailed({
          operation: "list skills",
          reason: cause instanceof Error ? cause.message : "skill listing failed",
        }),
    })
    return Array.from(skills.values()).map((skill) => ({
      name: skill.name,
      description: skill.description,
      path: skill.path,
    }))
  })
}

export function getSkill(
  cwd: string,
  name: string
): Effect.Effect<SkillContent, SessionError, FileSystem.FileSystem> {
  return Effect.gen(function* () {
    const skills = yield* Effect.tryPromise({
      try: () => loadSkills(cwd),
      catch: (cause) =>
        new SessionOperationFailed({
          operation: `get skill ${name}`,
          reason: cause instanceof Error ? cause.message : "skill load failed",
        }),
    })
    const skill = skills.get(name)
    if (!skill) {
      return yield* new SessionOperationFailed({
        operation: `get skill ${name}`,
        reason: `Skill not found: ${name}`,
      })
    }
    const fs = yield* FileSystem.FileSystem
    const content = yield* fs.readFileString(skill.path).pipe(
      Effect.mapError(
        (cause) =>
          new SessionOperationFailed({
            operation: `get skill ${name}`,
            reason: cause instanceof Error ? cause.message : "skill read failed",
          })
      )
    )
    return { name: skill.name, content }
  })
}

export function runBash(
  context: { cwd: string; projectRoot: string; scratchpadPath: string },
  command: string,
  stdin?: string | undefined
): Effect.Effect<RunBashResult, SessionError, CommandExecutor.CommandExecutor> {
  return Effect.scoped(
    Effect.gen(function* () {
      const shell = process.env.SHELL || "/bin/sh"
      const baseCmd = Command.make(shell, "-c", command).pipe(
        Command.workingDirectory(context.cwd),
        Command.env({
          ...process.env,
          PROJECT_ROOT: context.projectRoot,
          M: context.scratchpadPath,
        })
      )
      const cmd = stdin ? Command.feed(baseCmd, stdin) : baseCmd
      const proc = yield* Command.start(cmd)
      const stdout = yield* proc.stdout.pipe(
        Stream.decodeText(),
        Stream.runCollect,
        Effect.map((chunk) => Chunk.toReadonlyArray(chunk).join(""))
      )
      const stderr = yield* proc.stderr.pipe(
        Stream.decodeText(),
        Stream.runCollect,
        Effect.map((chunk) => Chunk.toReadonlyArray(chunk).join(""))
      )
      const exitCode = yield* proc.exitCode
      return {
        stdout,
        stderr,
        exitCode: Number(exitCode),
        cwd: context.cwd,
      }
    })
  ).pipe(
    Effect.mapError(
      (cause) =>
        new SessionOperationFailed({
          operation: "run bash",
          reason: cause instanceof Error ? cause.message : "bash failed",
        })
    )
  )
}
