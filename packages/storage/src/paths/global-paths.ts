import { Buffer } from 'node:buffer'
import { createHash } from 'node:crypto'
import { homedir } from 'node:os'
import { join, resolve } from 'node:path'

export interface GlobalStoragePaths {
  readonly root: string

  readonly configFile: string
  readonly authFile: string
  readonly modelCacheFile: string

  readonly sessionsRoot: string
  readonly indexRoot: string
  readonly pendingMemoryExtractionRoot: string

  readonly tracesRoot: string

  readonly logsRoot: string
  readonly cliLogFile: string
  readonly eventLogFile: string

  readonly skillsRoot: string

  readonly sessionDir: (sessionId: string) => string
  readonly sessionMetaFile: (sessionId: string) => string
  readonly sessionEventsFile: (sessionId: string) => string
  readonly sessionProjectionSnapshotFile: (sessionId: string) => string
  readonly sessionAddressedEntriesRoot: (sessionId: string) => string
  readonly sessionAddressedEntryFile: (
    sessionId: string,
    namespace: string,
    address: string
  ) => string
  readonly sessionLogFile: (sessionId: string) => string
  readonly sessionScratchpad: (sessionId: string) => string

  readonly cwdIndexFile: (cwd: string) => string
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

  const indexRoot = join(sessionsRoot, 'index')

  function hashCwd(cwd: string): string {
    return createHash('sha256').update(resolve(cwd)).digest('hex')
  }

  function encodeAddressedPathPart(value: string): string {
    return Buffer.from(value, 'utf8').toString('base64url')
  }

  return {
    root,

    configFile: join(root, 'config.json'),
    authFile: join(root, 'auth.json'),
    modelCacheFile: join(root, 'model-cache.json'),

    sessionsRoot,
    indexRoot,
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
    sessionProjectionSnapshotFile: (sessionId: string) =>
      join(sessionsRoot, sessionId, 'projections.json'),
    sessionAddressedEntriesRoot: (sessionId: string) =>
      join(sessionsRoot, sessionId, 'addressed'),
    sessionAddressedEntryFile: (
      sessionId: string,
      namespace: string,
      address: string
    ) =>
      join(
        sessionsRoot,
        sessionId,
        'addressed',
        encodeAddressedPathPart(namespace),
        `${encodeAddressedPathPart(address)}.json`
      ),
    sessionLogFile: (sessionId: string) =>
      join(sessionsRoot, sessionId, 'logs.jsonl'),
    sessionScratchpad: (sessionId: string) =>
      join(sessionsRoot, sessionId, 'scratchpad'),

    cwdIndexFile: (cwd: string) => join(indexRoot, `${hashCwd(cwd)}.json`),

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
