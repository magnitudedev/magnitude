/**
 * Recent tasks data layer
 *
 * Loads task summaries from .magnitude/tasks/ for display in the task panel empty state
 */

import * as fs from 'fs/promises'
import * as path from 'path'

export interface RecentTask {
  id: string
  title: string
  label: string | null
  status: string
  date: string // YY-MM-DD folder name
  updated: string
}

export interface PreviewedTask {
  id: string
  title: string
  label: string | null
  status: string
  details: string
  date: string
}

const MAX_RECENT_TASKS = 20

/**
 * Parse frontmatter from a markdown file to extract task summary info
 */
function parseFrontmatterQuick(content: string): Record<string, string> {
  const match = content.match(/^---\n([\s\S]*?)\n---/)
  if (!match) return {}
  const result: Record<string, string> = {}
  for (const line of match[1].split('\n')) {
    const colonIdx = line.indexOf(':')
    if (colonIdx > 0) {
      result[line.slice(0, colonIdx).trim()] = line.slice(colonIdx + 1).trim()
    }
  }
  return result
}

/**
 * Extract title from markdown body (first # heading)
 */
function extractTitle(content: string): string {
  const bodyMatch = content.match(/^---\n[\s\S]*?\n---\n([\s\S]*)/)
  const body = bodyMatch ? bodyMatch[1] : content
  const titleMatch = body.match(/^#\s+(.+)$/m)
  return titleMatch ? titleMatch[1] : 'Untitled'
}

/**
 * Get recent tasks from .magnitude/tasks/ directory
 */
export async function getRecentTasks(workingDirectory: string = process.cwd(), limit = MAX_RECENT_TASKS): Promise<RecentTask[]> {
  const tasksDir = path.join(workingDirectory, '.magnitude', 'tasks')

  try {
    await fs.access(tasksDir)
  } catch {
    return []
  }

  const tasks: RecentTask[] = []

  try {
    const entries = await fs.readdir(tasksDir, { withFileTypes: true })
    const dateFolders = entries
      .filter(e => e.isDirectory())
      .sort((a, b) => b.name.localeCompare(a.name)) // newest first

    for (const folder of dateFolders) {
      const folderPath = path.join(tasksDir, folder.name)
      const files = await fs.readdir(folderPath)

      for (const file of files) {
        if (!file.endsWith('.md')) continue

        try {
          const content = await fs.readFile(path.join(folderPath, file), 'utf-8')
          const frontmatter = parseFrontmatterQuick(content)
          const title = extractTitle(content)

          tasks.push({
            id: file.replace('.md', ''),
            title,
            label: frontmatter.label || null,
            status: frontmatter.status || 'draft',
            date: folder.name,
            updated: frontmatter.updated || frontmatter.created || '',
          })
        } catch {
          // Skip files that can't be read
          continue
        }
      }

      if (tasks.length >= limit) break
    }
  } catch {
    return []
  }

  // Sort by updated timestamp descending
  tasks.sort((a, b) => {
    if (a.updated && b.updated) return b.updated.localeCompare(a.updated)
    return b.date.localeCompare(a.date)
  })

  return tasks.slice(0, limit)
}

/**
 * Format a date folder name (YY-MM-DD) for display
 */
export function formatTaskDate(dateFolder: string): string {
  // dateFolder is like "26-02-15"
  const parts = dateFolder.split('-')
  if (parts.length !== 3) return dateFolder
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
  const monthIdx = parseInt(parts[1], 10) - 1
  const month = months[monthIdx] || parts[1]
  return `${month} ${parseInt(parts[2], 10)}`
}

/**
 * Load full task content from a .magnitude/tasks/ file by ID
 * Searches all date folders for the task
 */
export async function loadFullTask(taskId: string, workingDirectory: string = process.cwd()): Promise<PreviewedTask | null> {
  const tasksDir = path.join(workingDirectory, '.magnitude', 'tasks')

  try {
    await fs.access(tasksDir)
  } catch {
    return null
  }

  try {
    const entries = await fs.readdir(tasksDir, { withFileTypes: true })
    const dateFolders = entries.filter(e => e.isDirectory())

    for (const folder of dateFolders) {
      const filePath = path.join(tasksDir, folder.name, `${taskId}.md`)
      try {
        const content = await fs.readFile(filePath, 'utf-8')
        const frontmatter = parseFrontmatterQuick(content)
        const title = extractTitle(content)

        // Extract details (everything between ## Details and next ##)
        const bodyMatch = content.match(/^---\n[\s\S]*?\n---\n([\s\S]*)/)
        const body = bodyMatch ? bodyMatch[1] : content
        const detailsMatch = body.match(/## Details\n([\s\S]*?)(?=\n## |$)/)
        const details = detailsMatch ? detailsMatch[1].trim() : ''

        return {
          id: taskId,
          title,
          label: frontmatter.label || null,
          status: frontmatter.status || 'draft',
          details,
          date: folder.name,
        }
      } catch {
        continue
      }
    }
    return null
  } catch {
    return null
  }
}
