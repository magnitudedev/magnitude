// TODO: flesh out
/**
 * Represents any reason why a test case could have failed, for example:
 * - Step could not be completed
 * - Check did not pass
 * - Could not navigate to starting URL
 * - Time or action based timeout
 * - Operation cancelled by signal
 * - ...
 */
// export interface FailureDescriptor {
//     description: string
// }
export type FailureDescriptor = BugDetectedFailure | MisalignmentFailure | NetworkFailure | BrowserFailure | RateLimitFailure | ApiKeyFailure | UnknownFailure | CancelledFailure;

export type BugSeverity = 'critical' | 'high' | 'medium' | 'low';

/**
 * Step and check failures are classified into one of:
 * BugDetectedFailure: seems to be something wrong with the application itself
 * MisalignmentFailure: seems to be a discrepency between the test case and the interface
 *    OR the agent did not properly recognize the relationship between test case and interface
 */

export interface BugDetectedFailure {
    variant: 'bug'
    title: string
    expectedResult: string
    actualResult: string
    severity: BugSeverity
}

export interface MisalignmentFailure {
    /**
     * Major misalignment: when a step/check fails due to:
     * 1. Poorly written step/check that is completely unrelated to the interface
     * 2. Or interface has changed so much that step/check no longer applicable
     * 3. Planner did not do good enough job adjusting recipe for minor misalignment
     * Misalignment could be due to a poorly written test case OR bad agent behavior.
     */
    variant: 'misalignment',
    // Some message speculating about what may have gone wrong, ideally that would help user know how to adjust TC to fix
    message: string
}

export interface NetworkFailure {
    /**
     * For example, failure to connect to starting URL, or any other network errors
     * that would completely prevent the test from executing.
     */
    variant: 'network'
    message: string
}

export interface BrowserFailure {
    /**
     * E.g. something goes wrong with playwright interactions, any DOM manipulation, etc.
     */
    variant: 'browser'
    message: string
}

export interface RateLimitFailure {
    variant: 'rate_limit',
    message: string
}

export interface ApiKeyFailure {
    variant: 'api_key'
    message: string
}

export interface UnknownFailure {
    // Failure due to some unknown / unhandled error.
    // If these are being returned we should identify and handle them specifically
    variant: 'unknown'
    message: string
}

export interface CancelledFailure {
    /**
     * Operation was cancelled, typically by an AbortSignal from a controlling process (e.g., test runner pool).
     */
    variant: 'cancelled'
}

/**
 * Convert a failure variant to its human-readable title
 * @param variant The failure variant
 * @returns The human-readable title for the variant
 */
export function variantToTitle(variant: FailureDescriptor['variant']): string {
    const titles: Record<typeof variant, string> = {
        'bug': 'Bug detected',
        'misalignment': 'Misalignment detected',
        'cancelled': 'Operation cancelled',
        'network': 'Network failure',
        'browser': 'Browser failure',
        'api_key': 'API authentication error',
        'rate_limit': 'Rate limit hit',
        'unknown': 'Error'
    } as const;

    return titles[variant] || variant;
}

/**
 * Generate a simplified string representation of a failure suitable for Error messages
 * @param failure The failure descriptor
 * @returns A single string formatted for Error objects
 */
export function generateSimpleFailureString(failure: FailureDescriptor): string {
    const lines: string[] = [];

    const addSection = (header: string, content: string | string[]) =>
        lines.push(header,
            ...(Array.isArray(content)
                ? content.map((line) => `  ${line}`)
                : [`  ${content}`])
        );

    const addStandardError = (title: string, message?: string) =>
        lines.push(message ? `${title}: ${message}` : title);

    const variantTitle = variantToTitle(failure.variant)
    switch (failure.variant) {
        case 'bug':
            addStandardError(variantTitle, failure.title);
            addStandardError(variantToTitle(failure.variant));
            addSection('Expected:', failure.expectedResult);
            addSection('Actual:', failure.actualResult);
            lines.push(`Severity: ${failure.severity.toUpperCase()}`);
            break;

        case 'cancelled':
            addStandardError(variantTitle);
            break;

        case 'network':
        case 'browser':
        case 'misalignment':
        case 'api_key':
        case 'rate_limit':
            addStandardError(variantTitle, failure.message);
            break;

        case 'unknown':
        default:
            if ('message' in failure && failure.message) {
                addStandardError(variantTitle, failure.message);
            } else {
                lines.push(`Unknown error: ${failure.variant || 'unspecified'}`);
            }
            break;
    }

    return lines.join('\n');
}
