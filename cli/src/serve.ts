import { createStorageClient } from '@magnitudedev/storage'
import { SessionManager } from './serve/session-manager'
import { handleSessionsRoute } from './serve/routes/sessions'
import { handleEventsRoute } from './serve/routes/events'
import { handleHealthRoute } from './serve/routes/health'
import { authenticateRequest } from './serve/middleware/auth'

export interface ServeOptions {
  port: number
  host: string
  token?: string
  debug: boolean
}

const CORS_HEADERS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization, Last-Event-ID',
  'Access-Control-Allow-Methods': 'GET, POST, DELETE, OPTIONS',
}

function withCors(response: Response): Response {
  const headers = new Headers(response.headers)
  for (const [key, value] of Object.entries(CORS_HEADERS)) {
    headers.set(key, value)
  }
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  })
}

function errorJson(message: string, status = 500): Response {
  return new Response(JSON.stringify({ error: message }), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

export async function startServer(options: ServeOptions): Promise<void> {
  if (!Number.isFinite(options.port) || options.port < 1 || options.port > 65535) {
    throw new Error(`Invalid port: ${options.port}. Must be between 1 and 65535.`)
  }
  if (!options.host?.trim()) {
    throw new Error('Invalid host')
  }

  const storage = await createStorageClient({ cwd: process.cwd() })
  const sessionManager = new SessionManager({ debug: options.debug, storage })

  const server = Bun.serve({
    port: options.port,
    hostname: options.host,
    fetch: async (req) => {
      try {
        const url = new URL(req.url)

        if (req.method === 'OPTIONS') {
          return withCors(new Response(null, { status: 204, headers: CORS_HEADERS }))
        }

        if (url.pathname !== '/health') {
          const auth = authenticateRequest(req, options.token)
          if (!auth.ok) return withCors(auth.response)
        }

        let response: Response
        if (url.pathname === '/health') {
          response = handleHealthRoute(sessionManager)
        } else if (url.pathname === '/events' || /^\/sessions\/[^/]+\/events$/.test(url.pathname)) {
          response = await handleEventsRoute(req, url, sessionManager)
        } else if (url.pathname === '/sessions' || url.pathname.startsWith('/sessions/')) {
          response = await handleSessionsRoute(req, url, sessionManager)
        } else {
          response = errorJson('Not found', 404)
        }

        return withCors(response)
      } catch (error) {
        const message = options.debug && error instanceof Error ? error.message : 'Internal server error'
        return withCors(errorJson(message, 500))
      }
    },
  })

  let shuttingDown = false
  const shutdown = async () => {
    if (shuttingDown) return
    shuttingDown = true
    await sessionManager.disposeAll()
    server.stop(true)
    process.exit(0)
  }

  process.once('SIGINT', shutdown)
  process.once('SIGTERM', shutdown)

  console.log(`magnitude serve listening on http://${options.host}:${options.port}`)
  await new Promise<void>(() => {})
}