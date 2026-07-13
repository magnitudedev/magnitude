function normalizePlatform(platform: NodeJS.Platform): string {
  switch (platform) {
    case 'darwin': return 'macOS'
    case 'win32': return 'Windows'
    default: return 'Linux'
  }
}

function detectShell(): string {
  return process.env.SHELL?.split('/').pop() || 'bash'
}

export const CLIENT_PLATFORM = normalizePlatform(process.platform)
export const CLIENT_SHELL = detectShell()

export const HEADER_PLATFORM = 'x-magnitude-platform'
export const HEADER_SHELL = 'x-magnitude-shell'
export const HEADER_SESSION_ID = 'x-magnitude-session-id'
export const HEADER_USE_DEDICATED = 'x-magnitude-use-dedicated'
