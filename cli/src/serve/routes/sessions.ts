import type { SessionManager } from '../session-manager'
import { SessionNotFoundError } from '../session-manager'

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      'Content-Type': 'application/json',
    },
  })
}

function badRequest(message: string): Response {
  return json({ error: message }, 400)
}

type CreateSessionBody = { cwd?: unknown }

async function parseJsonBody<T>(req: Request): Promise<T> {
  try {
    return await req.json() as T
  } catch {
    throw new Error('Invalid JSON body')
  }
}

export async function handleSessionsRoute(req: Request, url: URL, manager: SessionManager): Promise<Response> {
  const path = url.pathname
  const parts = path.split('/').filter(Boolean)

  if (parts.length === 1 && parts[0] === 'sessions') {
    if (req.method === 'POST') {
      try {
        const contentType = req.headers.get('content-type') ?? ''
        let cwd: string | undefined

        if (contentType.toLowerCase().includes('application/json')) {
          const body = await parseJsonBody<CreateSessionBody>(req)
          if (body.cwd !== undefined && typeof body.cwd !== 'string') {
            return badRequest('cwd must be a string')
          }
          cwd = body.cwd
        }

        const session = await manager.createSession(cwd !== undefined ? { cwd } : undefined)
        return json(session, 201)
      } catch (error) {
        if (error instanceof Error && error.message === 'Invalid JSON body') {
          return badRequest('Invalid JSON body')
        }
        if (error instanceof Error && error.message.startsWith('Invalid cwd:')) {
          return badRequest(error.message)
        }
        throw error
      }
    }
    if (req.method === 'GET') {
      return json(manager.listSessions())
    }
    return json({ error: 'Not found' }, 404)
  }

  if (parts.length >= 2 && parts[0] === 'sessions') {
    const sessionId = parts[1]

    if (parts.length === 2) {
      if (req.method === 'GET') {
        const detail = manager.getSessionDetail(sessionId)
        if (!detail) return json({ error: 'Session not found' }, 404)
        return json(detail)
      }
      if (req.method === 'DELETE') {
        const deleted = await manager.deleteSession(sessionId)
        if (!deleted) return json({ error: 'Session not found' }, 404)
        return new Response(null, { status: 204 })
      }
      return json({ error: 'Not found' }, 404)
    }

    try {
      if (req.method === 'POST' && parts.length === 3 && parts[2] === 'messages') {
        const body = await parseJsonBody<{ content?: unknown }>(req)
        if (typeof body.content !== 'string') return badRequest('Missing content string')
        if (!body.content.trim()) return badRequest('Content cannot be empty')
        await manager.sendUserMessage(sessionId, body.content)
        return new Response(null, { status: 202 })
      }

      if (req.method === 'POST' && parts.length === 3 && parts[2] === 'interrupt') {
        await manager.interrupt(sessionId)
        return new Response(null, { status: 202 })
      }

      if (req.method === 'POST' && parts.length === 5 && parts[2] === 'tools' && parts[4] === 'approve') {
        const toolCallId = parts[3]
        await manager.approveTool(sessionId, toolCallId)
        return new Response(null, { status: 202 })
      }

      if (req.method === 'POST' && parts.length === 5 && parts[2] === 'tools' && parts[4] === 'reject') {
        const toolCallId = parts[3]
        let reason: string | undefined
        const contentType = req.headers.get('content-type') ?? ''
        if (contentType.toLowerCase().includes('application/json')) {
          const body = await parseJsonBody<{ reason?: unknown }>(req)
          if (body.reason !== undefined && typeof body.reason !== 'string') {
            return badRequest('reason must be a string')
          }
          reason = body.reason
        }
        await manager.rejectTool(sessionId, toolCallId, reason)
        return new Response(null, { status: 202 })
      }
    } catch (error) {
      if (error instanceof SessionNotFoundError) {
        return json({ error: 'Session not found' }, 404)
      }
      if (error instanceof Error && error.message === 'Invalid JSON body') {
        return badRequest('Invalid JSON body')
      }
      if (error instanceof Error && error.message.startsWith('Invalid cwd:')) {
        return badRequest(error.message)
      }
      throw error
    }
  }

  return json({ error: 'Not found' }, 404)
}