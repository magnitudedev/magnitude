import { join } from 'node:path'

export interface ProjectStoragePaths {
  readonly cwd: string
  readonly root: string

  readonly memoryFile: string
  readonly tasksDir: string
  readonly taskDateDir: (date: string) => string
  readonly taskFile: (date: string, taskId: string) => string

  readonly skillsRoot: string
  readonly projectSkillDir: (skillName: string) => string
  readonly projectSkillFile: (skillName: string) => string
}

export function makeProjectStoragePaths(cwd: string): ProjectStoragePaths {
  const root = join(cwd, '.magnitude')
  const skillsRoot = join(root, 'skills')

  return {
    cwd,
    root,
    memoryFile: join(root, 'memory.md'),
    tasksDir: join(root, 'tasks'),
    taskDateDir: (date: string) => join(root, 'tasks', date),
    taskFile: (date: string, taskId: string) =>
      join(root, 'tasks', date, `${taskId}.md`),
    skillsRoot,
    projectSkillDir: (skillName: string) => join(skillsRoot, skillName),
    projectSkillFile: (skillName: string) =>
      join(skillsRoot, skillName, 'SKILL.md'),
  }
}