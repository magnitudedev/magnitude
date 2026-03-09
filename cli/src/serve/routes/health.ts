import type { SessionManager } from '../session-manager'

export function handleHealthRoute(manager: SessionManager): Response {
  return new Response(
    JSON.stringify({
      status: 'ok',
      sessions: manager.listSessions().length,
    }),
    {
      status: 200,
      headers: {
        'Content-Type': 'application/json',
      },
    }
  )
}