import { readdir, readFile, stat } from 'fs/promises'
import { join } from 'path'
import { homedir } from 'os'
import { watch } from 'fs'

const TRACES_DIR = join(homedir(), '.magnitude', 'traces')
const PORT = 4776
const DIST_DIR = join(import.meta.dir, '..', 'dist')

// --- Helpers ---

async function getSessionDirs(): Promise<{ id: string; timestamp: string; meta?: Record<string, any> }[]> {
  try {
    const entries = await readdir(TRACES_DIR, { withFileTypes: true })
    const sessions = []
    for (const entry of entries) {
      if (!entry.isDirectory()) continue
      const sessionPath = join(TRACES_DIR, entry.name)
      let meta: Record<string, any> | undefined
      try {
        meta = JSON.parse(await readFile(join(sessionPath, 'meta.json'), 'utf-8'))
      } catch {}
      sessions.push({
        id: entry.name,
        timestamp: meta?.created ?? entry.name,
        meta,
      })
    }
    sessions.sort((a, b) => b.id.localeCompare(a.id))
    return sessions
  } catch {
    return []
  }
}

async function getSessionTraces(sessionId: string): Promise<any[]> {
  const tracesFile = join(TRACES_DIR, sessionId, 'traces.jsonl')
  try {
    const content = await readFile(tracesFile, 'utf-8')
    return content.trim().split('\n').filter(Boolean).map(line => {
      try { return JSON.parse(line) } catch { return null }
    }).filter(Boolean)
  } catch {
    return []
  }
}

// --- Server ---

const server = Bun.serve({
  port: PORT,
  async fetch(req) {
    const url = new URL(req.url)
    const path = url.pathname

    // CORS
    const corsHeaders = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    }

    if (req.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders })
    }

    // API routes
    if (path === '/api/sessions') {
      const sessions = await getSessionDirs()
      const rawLimit = Number(url.searchParams.get('limit') ?? '50')
      const limit = Math.max(1, Math.min(200, Number.isFinite(rawLimit) ? rawLimit : 50))
      const cursor = url.searchParams.get('cursor')

      let startIndex = 0
      if (cursor) {
        const cursorIndex = sessions.findIndex((s) => s.id === cursor)
        startIndex = cursorIndex >= 0 ? cursorIndex + 1 : 0
      }

      const items = sessions.slice(startIndex, startIndex + limit)
      const nextCursor = startIndex + limit < sessions.length && items.length > 0
        ? items[items.length - 1].id
        : null

      return Response.json({ items, nextCursor }, { headers: corsHeaders })
    }

    const sessionMatch = path.match(/^\/api\/sessions\/([^/]+)$/)
    if (sessionMatch) {
      const sessionId = decodeURIComponent(sessionMatch[1])
      const metaFile = join(TRACES_DIR, sessionId, 'meta.json')
      try {
        const meta = JSON.parse(await readFile(metaFile, 'utf-8'))
        return Response.json(meta, { headers: corsHeaders })
      } catch {
        return Response.json({}, { headers: corsHeaders })
      }
    }

    const tracesMatch = path.match(/^\/api\/sessions\/([^/]+)\/traces$/)
    if (tracesMatch) {
      const sessionId = decodeURIComponent(tracesMatch[1])
      const traces = await getSessionTraces(sessionId)
      return Response.json(traces, { headers: corsHeaders })
    }

    const streamMatch = path.match(/^\/api\/sessions\/([^/]+)\/stream$/)
    if (streamMatch) {
      const sessionId = decodeURIComponent(streamMatch[1])
      const tracesFile = join(TRACES_DIR, sessionId, 'traces.jsonl')

      // SSE: watch file for changes and stream new lines
      let lastSize = 0
      try {
        const s = await stat(tracesFile)
        lastSize = s.size
      } catch {}

      const stream = new ReadableStream({
        start(controller) {
          controller.enqueue('retry: 1000\n\n')

          const watcher = watch(tracesFile, async () => {
            try {
              const s = await stat(tracesFile)
              if (s.size > lastSize) {
                const fd = Bun.file(tracesFile)
                const content = await fd.text()
                const lines = content.split('\n').filter(Boolean)
                const oldContent = content.slice(0, lastSize)
                const oldLineCount = oldContent.split('\n').filter(Boolean).length
                const newLines = lines.slice(oldLineCount)
                lastSize = s.size
                for (const line of newLines) {
                  try {
                    JSON.parse(line) // validate
                    controller.enqueue(`data: ${line}\n\n`)
                  } catch {}
                }
              }
            } catch {}
          })

          req.signal.addEventListener('abort', () => {
            watcher.close()
            controller.close()
          })
        },
      })

      return new Response(stream, {
        headers: {
          ...corsHeaders,
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive',
        },
      })
    }

    // Static file serving (built Svelte app)
    try {
      let filePath = path === '/' ? '/index.html' : path
      const file = Bun.file(join(DIST_DIR, filePath))
      if (await file.exists()) {
        return new Response(file)
      }
      // SPA fallback
      const index = Bun.file(join(DIST_DIR, 'index.html'))
      if (await index.exists()) {
        return new Response(index)
      }
    } catch {}

    return new Response('Not Found', { status: 404 })
  },
})

console.log(`Magnitude Traces dashboard running at http://localhost:${PORT}`)
