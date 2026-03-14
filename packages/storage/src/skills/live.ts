import { Effect, Layer } from 'effect'

import { ProjectStorage } from '../services'
import { SkillStorage } from './contracts'
import {
  ensureProjectSkillsDir,
  listProjectSkills,
  readProjectSkill,
  removeProjectSkill,
  writeProjectSkill,
} from './storage'

export const SkillStorageLive = Layer.effect(
  SkillStorage,
  Effect.gen(function* () {
    const projectStorage = yield* ProjectStorage

    return SkillStorage.of({
      ensureDir: () =>
        Effect.promise(() => ensureProjectSkillsDir(projectStorage.paths)),
      list: () => Effect.promise(() => listProjectSkills(projectStorage.paths)),
      read: (skillName: string) =>
        Effect.promise(() => readProjectSkill(projectStorage.paths, skillName)),
      write: (skillName: string, content: string) =>
        Effect.promise(() =>
          writeProjectSkill(projectStorage.paths, skillName, content)
        ),
      remove: (skillName: string) =>
        Effect.promise(() => removeProjectSkill(projectStorage.paths, skillName)),
    })
  })
)