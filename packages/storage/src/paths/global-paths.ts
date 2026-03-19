import { homedir } from 'node:os'
import { join } from 'node:path'

export interface GlobalStoragePaths {
  readonly root: string

  readonly configFile: string
  readonly authFile: string

  readonly sessionsRoot: string
  readonly pendingMemoryExtractionRoot: string

  readonly tracesRoot: string

  readonly logsRoot: string
  readonly cliLogFile: string
  readonly eventLogFile: string

  readonly skillsRoot: string

  readonly sessionDir: (sessionId: string) => string
  readonly sessionMetaFile: (sessionId: string) => string
  readonly sessionEventsFile: (sessionId: string) => string
  readonly sessionLogFile: (sessionId: string) => string
  readonly sessionWorkspace: (sessionId: string) => string
  readonly sessionArtifactsDir: (sessionId: string) => string
  readonly sessionArtifactFile: (sessionId: string, artifactName: string) => string

  readonly pendingMemoryJobFile: (jobId: string) => string

  readonly traceDir: (traceId: string) => string
  readonly traceMetaFile: (traceId: string) => string
  readonly traceEventsFile: (traceId: string) => string

  readonly globalSkillDir: (skillName: string) => string
  readonly globalSkillFile: (skillName: string) => string
}

export function defaultGlobalStorageRoot(): string {
  return join(homedir(), '.magnitude')
}

export function makeGlobalStoragePaths(root: string): GlobalStoragePaths {
  const sessionsRoot = join(root, 'sessions')
  const tracesRoot = join(root, 'traces')
  const logsRoot = join(root, 'logs')
  const skillsRoot = join(root, 'skills')
  const pendingMemoryExtractionRoot = join(
    sessionsRoot,
    '.pending-memory-extraction'
  )

  return {
    root,

    configFile: join(root, 'config.json'),
    authFile: join(root, 'auth.json'),

    sessionsRoot,
    pendingMemoryExtractionRoot,

    tracesRoot,

    logsRoot,
    cliLogFile: join(logsRoot, 'cli.jsonl'),
    eventLogFile: join(logsRoot, 'events.jsonl'),

    skillsRoot,

    sessionDir: (sessionId: string) => join(sessionsRoot, sessionId),
    sessionMetaFile: (sessionId: string) =>
      join(sessionsRoot, sessionId, 'meta.json'),
    sessionEventsFile: (sessionId: string) =>
      join(sessionsRoot, sessionId, 'events.jsonl'),
    sessionLogFile: (sessionId: string) =>
      join(sessionsRoot, sessionId, 'logs.jsonl'),
    sessionWorkspace: (sessionId: string) =>
      join(sessionsRoot, sessionId, 'workspace'),
    sessionArtifactsDir: (sessionId: string) =>
      join(sessionsRoot, sessionId, 'artifacts'),
    sessionArtifactFile: (sessionId: string, artifactName: string) =>
      join(sessionsRoot, sessionId, 'artifacts', artifactName),

    pendingMemoryJobFile: (jobId: string) =>
      join(pendingMemoryExtractionRoot, `${jobId}.json`),

    traceDir: (traceId: string) => join(tracesRoot, traceId),
    traceMetaFile: (traceId: string) => join(tracesRoot, traceId, 'meta.json'),
    traceEventsFile: (traceId: string) =>
      join(tracesRoot, traceId, 'traces.jsonl'),

    globalSkillDir: (skillName: string) => join(skillsRoot, skillName),
    globalSkillFile: (skillName: string) =>
      join(skillsRoot, skillName, 'SKILL.md'),
  }
}