import { Effect } from "effect"
import * as FileSystem from "@effect/platform/FileSystem"
import { loadSkills } from "@magnitudedev/skills"
import {
  OperationFailed,
  type SkillContent,
  type SkillListEntry,
} from "@magnitudedev/acn-protocol"

export function listSkills(
  cwd: string
): Effect.Effect<Array<SkillListEntry>, OperationFailed, FileSystem.FileSystem> {
  return Effect.gen(function* () {
    const skills = yield* Effect.tryPromise({
      try: () => loadSkills(cwd),
      catch: (cause) =>
        new OperationFailed({
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
): Effect.Effect<SkillContent, OperationFailed, FileSystem.FileSystem> {
  return Effect.gen(function* () {
    const skills = yield* Effect.tryPromise({
      try: () => loadSkills(cwd),
      catch: (cause) =>
        new OperationFailed({
          operation: `get skill ${name}`,
          reason: cause instanceof Error ? cause.message : "skill load failed",
        }),
    })
    const skill = skills.get(name)
    if (!skill) {
      return yield* new OperationFailed({
        operation: `get skill ${name}`,
        reason: `Skill not found: ${name}`,
      })
    }
    const fs = yield* FileSystem.FileSystem
    const content = yield* fs.readFileString(skill.path).pipe(
      Effect.mapError(
        (cause) =>
          new OperationFailed({
            operation: `get skill ${name}`,
            reason: cause instanceof Error ? cause.message : "skill read failed",
          })
      )
    )
    return { name: skill.name, content }
  })
}
