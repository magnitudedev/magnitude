import {
  ensureDir,
  listDirectory,
  pathExists,
  readTextFile,
  removeFileIfExists,
  writeTextFile,
} from '../io'
import type { ProjectStoragePaths } from '../paths'

export function assertValidSkillName(skillName: string): void {
  if (
    skillName.length === 0 ||
    skillName === '.' ||
    skillName === '..' ||
    skillName.includes('/') ||
    skillName.includes('\\')
  ) {
    throw new Error(`Invalid skill name: ${skillName}`)
  }
}

export async function ensureProjectSkillsDir(
  paths: ProjectStoragePaths
): Promise<string> {
  await ensureDir(paths.skillsRoot)
  return paths.skillsRoot
}

export async function listProjectSkills(
  paths: ProjectStoragePaths
): Promise<string[]> {
  try {
    const entries = await listDirectory(paths.skillsRoot)
    const skillNames = await Promise.all(
      entries
        .filter((entry) => entry.isDirectory)
        .map(async (entry) =>
          (await pathExists(paths.projectSkillFile(entry.name))) ? entry.name : null
        )
    )

    return skillNames.filter((skillName): skillName is string => skillName !== null)
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      return []
    }

    throw error
  }
}

export async function readProjectSkill(
  paths: ProjectStoragePaths,
  skillName: string
): Promise<string> {
  assertValidSkillName(skillName)
  return readTextFile(paths.projectSkillFile(skillName))
}

export async function writeProjectSkill(
  paths: ProjectStoragePaths,
  skillName: string,
  content: string
): Promise<void> {
  assertValidSkillName(skillName)
  await writeTextFile(paths.projectSkillFile(skillName), content)
}

export async function removeProjectSkill(
  paths: ProjectStoragePaths,
  skillName: string
): Promise<void> {
  assertValidSkillName(skillName)
  await removeFileIfExists(paths.projectSkillFile(skillName))
}