/**
 * Background process constants
 */

/** Default shell timeout when no per-invocation timeout is specified */
export const DEFAULT_TIMEOUT_S = 10

/** Extra grace period before auto-killing a timed-out detached process */
export const AUTO_KILL_BUFFER_S = 5

/** Delay before detaching explicitly backgrounded shell commands */
export const BACKGROUND_DETACH_MS = 1000

/** Character count threshold that triggers demotion to file mode */
export const DEMOTION_THRESHOLD_CHARS = 8192

/** Max characters in a line-aligned tail for demoted output */
export const TAIL_MAX_CHARS = 4096

/** Grace period before SIGKILL after SIGTERM on shutdown */
export const SHUTDOWN_GRACE_MS = 2_000

/** Directory for demoted process output files */
export const OUTPUT_DIR_NAME = '.magnitude/tmp'