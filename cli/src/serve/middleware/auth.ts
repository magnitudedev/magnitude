export function authenticateRequest(req: Request, token?: string): { ok: true } | { ok: false; response: Response } {
  const expectedToken = token?.trim()
  if (!expectedToken) return { ok: true }

  const authHeader = req.headers.get('authorization')
  if (!authHeader) {
    return {
      ok: false,
      response: new Response(JSON.stringify({ error: 'Unauthorized' }), {
        status: 401,
        headers: {
          'Content-Type': 'application/json',
          'WWW-Authenticate': 'Bearer',
        },
      }),
    }
  }

  const match = authHeader.match(/^Bearer\s+(.+)$/i)
  if (!match || match[1] !== expectedToken) {
    return {
      ok: false,
      response: new Response(JSON.stringify({ error: 'Unauthorized' }), {
        status: 401,
        headers: {
          'Content-Type': 'application/json',
          'WWW-Authenticate': 'Bearer',
        },
      }),
    }
  }

  return { ok: true }
}