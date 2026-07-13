import { readdir, readFile, rm } from 'node:fs/promises'
import { dirname, join, resolve } from 'node:path'
import { homedir } from 'node:os'
import { Schema } from 'effect'
import {
  AcnVersionRegistryJson,
  type AcnRegistration,
} from '@magnitudedev/protocol'
import type { AcnInfo, KillAllAcnResult, RpcTraceSummary } from './lib/types'

const PORT = Number(process.env.ACN_DASH_API_PORT ?? 4886)
const MOTEL_URL = process.env.MAGNITUDE_MOTEL_URL ?? 'http://127.0.0.1:27686'
const DATA_DIR = join(homedir(), '.magnitude')
const ACN_DIR = join(DATA_DIR, 'acn')
const DIST_DIR = join(import.meta.dir, '..', 'dist')
const decodeRegistry = Schema.decodeUnknownSync(AcnVersionRegistryJson)

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
}

function json(body: unknown, init?: ResponseInit): Response {
  return Response.json(body, {
    ...init,
    headers: {
      ...corsHeaders,
      ...(init?.headers ?? {}),
    },
  })
}

async function readRegistration(path: string): Promise<AcnRegistration | null> {
  try {
    const text = await readFile(path, 'utf8')
    if (text.trim().length === 0) return null
    return decodeRegistry(text).registration
  } catch {
    return null
  }
}

function isMissingProcess(error: unknown): boolean {
  return typeof error === 'object' &&
    error !== null &&
    'code' in error &&
    Reflect.get(error, 'code') === 'ESRCH'
}

async function removeStaleRegistration(registryPath: string): Promise<void> {
  await rm(registryPath, { force: true })
  await rm(dirname(registryPath), { recursive: false }).catch(() => undefined)
}

async function fetchJson(url: string, timeoutMs = 800): Promise<unknown> {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const response = await fetch(url, { signal: controller.signal })
    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText}`)
    }
    return await response.json()
  } finally {
    clearTimeout(timeout)
  }
}

interface MotelTraceSummary {
  readonly traceId?: unknown
  readonly serviceName?: unknown
  readonly rootOperationName?: unknown
  readonly startedAt?: unknown
  readonly isRunning?: unknown
  readonly durationMs?: unknown
  readonly spanCount?: unknown
  readonly errorCount?: unknown
  readonly warnings?: unknown
}

const numberValue = (value: unknown): number =>
  typeof value === 'number' && Number.isFinite(value) ? value : 0

const rpcKind = (rpcName: string): RpcTraceSummary['kind'] =>
  rpcName.startsWith('Stream') || rpcName.startsWith('Watch') ? 'stream' : 'command'

function toRpcTraceSummary(trace: MotelTraceSummary): RpcTraceSummary | null {
  if (
    typeof trace.traceId !== 'string' ||
    typeof trace.serviceName !== 'string' ||
    typeof trace.rootOperationName !== 'string' ||
    typeof trace.startedAt !== 'string' ||
    !trace.rootOperationName.startsWith('RpcServer.')
  ) {
    return null
  }

  const rpcName = trace.rootOperationName.slice('RpcServer.'.length)
  return {
    traceId: trace.traceId,
    serviceName: trace.serviceName,
    rootOperationName: trace.rootOperationName,
    startedAt: trace.startedAt,
    isRunning: trace.isRunning === true,
    durationMs: numberValue(trace.durationMs),
    spanCount: numberValue(trace.spanCount),
    errorCount: numberValue(trace.errorCount),
    warnings: Array.isArray(trace.warnings) ? trace.warnings.filter((warning): warning is string => typeof warning === 'string') : [],
    rpcName,
    kind: rpcKind(rpcName),
  }
}

async function listRpcTraces(): Promise<RpcTraceSummary[]> {
  const url = new URL('/api/traces/search', MOTEL_URL)
  url.searchParams.set('service', 'magnitude-acn')
  url.searchParams.set('operation', 'RpcServer.')
  url.searchParams.set('lookback', '4h')
  url.searchParams.set('limit', '40')

  const payload = await fetchJson(url.toString(), 1200) as { data?: MotelTraceSummary[] }
  return (payload.data ?? [])
    .map(toRpcTraceSummary)
    .filter((trace): trace is RpcTraceSummary => trace !== null)
}

async function probeAcn(registration: AcnRegistration, registryPath: string): Promise<AcnInfo> {
  let health: AcnInfo['health']
  try {
    const payload = await fetchJson(`${registration.url}/health`)
    const value = payload as {
      service?: string
      version?: string
      pid?: number
      schedulerElapsedMs?: number
    }
    health = {
      ok: value.service === 'magnitude-acn',
      service: value.service,
      version: value.version,
      pid: value.pid,
      schedulerElapsedMs: value.schedulerElapsedMs,
    }
  } catch (error) {
    health = {
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    }
  }

  let introspection: AcnInfo['introspection']
  try {
    await fetchJson(`${registration.url}/dev/introspection`)
    introspection = { ok: true }
  } catch (error) {
    introspection = {
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    }
  }

  return {
    version: registration.version,
    registration,
    registryPath,
    health,
    introspection,
  }
}

async function listAcns(): Promise<AcnInfo[]> {
  const registrations = await listRegistrations()

  const infos = await Promise.all(
    registrations.map(({ registration, registryPath }) => probeAcn(registration, registryPath)),
  )
  return infos.sort((a, b) => b.registration.timestamp - a.registration.timestamp)
}

async function listRegistrations(): Promise<Array<{ registration: AcnRegistration; registryPath: string }>> {
  let entries: string[] = []
  try {
    entries = await readdir(ACN_DIR)
  } catch {
    return []
  }

  const registrations: Array<{ registration: AcnRegistration; registryPath: string }> = []
  for (const entry of entries) {
    const registryPath = join(ACN_DIR, entry, 'registry.json')
    const registration = await readRegistration(registryPath)
    if (registration) registrations.push({ registration, registryPath })
  }

  return registrations
}

async function killAllAcns(): Promise<KillAllAcnResult[]> {
  const registrations = await listRegistrations()
  const results: KillAllAcnResult[] = []

  for (const { registration, registryPath } of registrations) {
    if (registration.pid === process.pid) {
      results.push({
        version: registration.version,
        pid: registration.pid,
        status: 'skipped_self',
      })
      continue
    }

    try {
      process.kill(registration.pid, 'SIGTERM')
      results.push({
        version: registration.version,
        pid: registration.pid,
        status: 'killed',
      })
    } catch (error) {
      if (isMissingProcess(error)) {
        await removeStaleRegistration(registryPath)
        results.push({
          version: registration.version,
          pid: registration.pid,
          status: 'stale',
        })
      } else {
        results.push({
          version: registration.version,
          pid: registration.pid,
          status: 'failed',
          error: error instanceof Error ? error.message : String(error),
        })
      }
    }
  }

  return results
}

async function findAcn(version: string): Promise<AcnInfo | null> {
  const acns = await listAcns()
  return acns.find((acn) => acn.version === version) ?? null
}

function upstreamUrl(acn: AcnInfo, suffix: string): string {
  return `${acn.registration.url}${suffix}`
}

async function proxyJson(version: string, suffix: string): Promise<Response> {
  const acn = await findAcn(version)
  if (!acn) return json({ error: 'not_found', message: `No ACN registration for ${version}` }, { status: 404 })
  if (!acn.introspection.ok) {
    return json({
      error: 'introspection_unavailable',
      message: acn.introspection.error ?? 'ACN introspection routes are not enabled',
    }, { status: 404 })
  }

  const response = await fetch(upstreamUrl(acn, suffix))
  const body = await response.text()
  return new Response(body, {
    status: response.status,
    statusText: response.statusText,
    headers: {
      ...corsHeaders,
      'Content-Type': response.headers.get('Content-Type') ?? 'application/json',
    },
  })
}

async function proxyStream(version: string, suffix: string): Promise<Response> {
  const acn = await findAcn(version)
  if (!acn) return json({ error: 'not_found', message: `No ACN registration for ${version}` }, { status: 404 })
  if (!acn.introspection.ok) {
    return json({
      error: 'introspection_unavailable',
      message: acn.introspection.error ?? 'ACN introspection routes are not enabled',
    }, { status: 404 })
  }

  const response = await fetch(upstreamUrl(acn, suffix))
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers: {
      ...corsHeaders,
      'Content-Type': response.headers.get('Content-Type') ?? 'text/event-stream',
      'Cache-Control': 'no-cache',
    },
  })
}

const server = Bun.serve({
  port: PORT,
  hostname: '127.0.0.1',
  async fetch(req) {
    const url = new URL(req.url)
    const path = url.pathname

    if (req.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders })
    }

    if (path === '/api/acns') {
      return json({ acns: await listAcns(), timestamp: Date.now() })
    }

    if (path === '/api/acns/kill-all' && req.method === 'POST') {
      const results = await killAllAcns()
      return json({ results, timestamp: Date.now() })
    }

    if (path === '/api/rpc-traces') {
      try {
        return json({ traces: await listRpcTraces(), timestamp: Date.now() })
      } catch (error) {
        return json({
          traces: [],
          timestamp: Date.now(),
          error: error instanceof Error ? error.message : String(error),
        })
      }
    }

    const sessionsMatch = path.match(/^\/api\/acns\/([^/]+)\/sessions$/)
    if (sessionsMatch) {
      const version = decodeURIComponent(sessionsMatch[1])
      return proxyJson(version, '/dev/sessions')
    }

    const introspectionMatch = path.match(/^\/api\/acns\/([^/]+)\/sessions\/([^/]+)\/introspection$/)
    if (introspectionMatch) {
      const version = decodeURIComponent(introspectionMatch[1])
      const sessionId = encodeURIComponent(decodeURIComponent(introspectionMatch[2]))
      const forkId = url.searchParams.get('forkId')
      const suffix = `/dev/sessions/${sessionId}${forkId ? `?forkId=${encodeURIComponent(forkId)}` : ''}`
      return proxyJson(version, suffix)
    }

    const streamMatch = path.match(/^\/api\/acns\/([^/]+)\/sessions\/([^/]+)\/stream$/)
    if (streamMatch) {
      const version = decodeURIComponent(streamMatch[1])
      const sessionId = encodeURIComponent(decodeURIComponent(streamMatch[2]))
      const forkId = url.searchParams.get('forkId')
      const suffix = `/dev/sessions/${sessionId}/stream${forkId ? `?forkId=${encodeURIComponent(forkId)}` : ''}`
      return proxyStream(version, suffix)
    }

    try {
      const filePath = path === '/' ? '/index.html' : path
      const resolved = resolve(DIST_DIR, '.' + filePath)
      if (!resolved.startsWith(DIST_DIR)) return new Response('Forbidden', { status: 403 })
      const file = Bun.file(resolved)
      if (await file.exists()) return new Response(file)
      const index = Bun.file(join(DIST_DIR, 'index.html'))
      if (await index.exists()) return new Response(index)
    } catch {}

    return new Response('Not Found', { status: 404 })
  },
})

console.log(`ACN dashboard API running at http://localhost:${server.port}`)
