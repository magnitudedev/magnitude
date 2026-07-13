/**
 * Recent tasks data layer
 *
 * Previously loaded task summaries from .magnitude/tasks/ for display in the
 * task panel empty state. In the client-server architecture, the client must
 * not touch the filesystem directly. Until a server-side RPC exists for task
 * listing, the empty state is returned.
 */

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

/**
 * Get recent tasks from .magnitude/tasks/ directory
 *
 * TODO: Replace with a server-side RPC once available. The client should not
 * read the filesystem directly.
 */
export async function getRecentTasks(_workingDirectory?: string, _limit?: number): Promise<RecentTask[]> {
  return []
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
 *
 * TODO: Replace with a server-side RPC once available. The client should not
 * read the filesystem directly.
 */
export async function loadFullTask(_taskId: string, _workingDirectory?: string): Promise<PreviewedTask | null> {
  return null
}
