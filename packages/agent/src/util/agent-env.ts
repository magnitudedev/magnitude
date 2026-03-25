/**
 * Standard environment for spawning shell commands in the agent context.
 * Ensures $M and $PROJECT_ROOT are available to all spawned processes.
 */
export function agentEnv(cwd: string, workspacePath: string): Record<string, string> {
  return {
    ...(process.env as Record<string, string>),
    NO_COLOR: '1',
    PROJECT_ROOT: cwd,
    M: workspacePath,
  }
}
