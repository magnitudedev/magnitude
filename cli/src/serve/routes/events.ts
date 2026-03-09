import type { SessionManager } from '../session-manager'
import { SessionNotFoundError } from '../session-manager'

function sseFrame(id: string, event: string, data: unknown): string {
  return `id: ${id}\nevent: ${event}\ndata: ${JSON.stringify(data)}\n\n`
}

export async function handleEventsRoute(req: Request, url: URL, manager: SessionManager): Promise<Response> {
  if (req.method !== 'GET') {
    return new Response(JSON.stringify({ error: 'Method not allowed' }), {
      status: 405,
      headers: { 'Content-Type': 'application/json' },
    })
  }

  const lastEventId = req.headers.get('Last-Event-ID')
  const parts = url.pathname.split('/').filter(Boolean)
  const encoder = new TextEncoder()

  if (parts.length === 1 && parts[0] === 'events') {
    let cleanup: (() => void) | null = null
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        const sub = manager.subscribeGlobalEvents((evt) => {
          controller.enqueue(encoder.encode(sseFrame(evt.id, evt.event, { sessionId: evt.sessionId, data: evt.data })))
        }, lastEventId)

        for (const evt of sub.replay) {
          controller.enqueue(encoder.encode(sseFrame(evt.id, evt.event, { sessionId: evt.sessionId, data: evt.data })))
        }

        const heartbeat = setInterval(() => {
          controller.enqueue(encoder.encode(': heartbeat\n\n'))
        }, 5000)

        cleanup = () => {
          clearInterval(heartbeat)
          sub.unsubscribe()
          cleanup = null
        }

        req.signal.addEventListener('abort', () => cleanup?.(), { once: true })
      },
      cancel() {
        cleanup?.()
      },
    })

    return new Response(stream, {
      status: 200,
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache, no-transform',
        Connection: 'keep-alive',
      },
    })
  }

  if (parts.length === 3 && parts[0] === 'sessions' && parts[2] === 'events') {
    const sessionId = parts[1]

    try {
      let cleanup: (() => void) | null = null
      const stream = new ReadableStream<Uint8Array>({
        start(controller) {
          const sub = manager.subscribeSessionEvents(sessionId, (evt) => {
            controller.enqueue(encoder.encode(sseFrame(evt.id, evt.event, evt.data)))
          }, lastEventId)

          for (const evt of sub.replay) {
            controller.enqueue(encoder.encode(sseFrame(evt.id, evt.event, evt.data)))
          }

          const heartbeat = setInterval(() => {
            controller.enqueue(encoder.encode(': heartbeat\n\n'))
          }, 5000)

          cleanup = () => {
            clearInterval(heartbeat)
            sub.unsubscribe()
            cleanup = null
          }

          req.signal.addEventListener('abort', () => cleanup?.(), { once: true })
        },
        cancel() {
          cleanup?.()
        },
      })

      return new Response(stream, {
        status: 200,
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache, no-transform',
          Connection: 'keep-alive',
        },
      })
    } catch (error) {
      if (error instanceof SessionNotFoundError) {
        return new Response(JSON.stringify({ error: 'Session not found' }), {
          status: 404,
          headers: { 'Content-Type': 'application/json' },
        })
      }
      throw error
    }
  }

  return new Response(JSON.stringify({ error: 'Not found' }), {
    status: 404,
    headers: { 'Content-Type': 'application/json' },
  })
}