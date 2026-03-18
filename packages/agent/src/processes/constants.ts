/**
 * Background process constants
 */

/** How long to wait before detaching a shell command as a background process */
export const DETACH_AFTER_MS = 5_000


/** Character count threshold that triggers demotion to file mode */
export const DEMOTION_THRESHOLD_CHARS = 8192

/** Max characters in a line-aligned tail for demoted output */
export const TAIL_MAX_CHARS = 4096

/** Grace period before SIGKILL after SIGTERM on shutdown */
export const SHUTDOWN_GRACE_MS = 2_000

/** Directory for demoted process output files */
export const OUTPUT_DIR_NAME = '.magnitude/tmp'