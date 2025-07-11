import logger from "@/logger";

export interface RetryOptions {
    errorSubstrings: string[],
    retryLimit: number,
    delayMs: number,
    warn: boolean
}

// export type RetryOptions =
//     ({ errorSubstrings: string[] } | { retryAll: true }) &
//     {retryLimit: number,
//     delayMs: number,
//     warn: boolean
// }

// export const DEFAULT_RETRY_OPTIONS = {

// }

export async function retryOnError<T>(
    fnToRetry: () => Promise<T>,
    { errorSubstrings, retryLimit, delayMs, warn }: RetryOptions
): Promise<T> {
    let lastError: any;

    if (retryLimit < 0) {
        retryLimit = 0;
    }

    for (let attempt = 0; attempt <= retryLimit; attempt++) {
        try {
            return await fnToRetry();
        } catch (error: any) {
            lastError = error;

            const errorMessage = String(error?.message ?? error);

            const includesSubstring = errorSubstrings.some((substring) => errorMessage.includes(substring));

            if (includesSubstring) {
                if (warn) {
                    logger.warn(`Retrying on: ${errorMessage}`);
                }
                if (attempt === retryLimit) {
                    throw lastError;
                }
            } else {
                // Error message does NOT contain the target substring. This error is not retryable.
                throw lastError; // Throw this current error immediately.
            }
        }
        await new Promise((resolve) => setTimeout(resolve, delayMs));
    }

    throw lastError;
}

/**
 * Performs a deep comparison between two values to determine if they are equivalent.
 *
 * @param a The first value to compare.
 * @param b The second value to compare.
 * @param cache A WeakMap to handle circular references. Should not be provided by the caller.
 * @returns `true` if the values are deeply equal, `false` otherwise.
 */
export function deepEquals(a: any, b: any, cache = new WeakMap<object, WeakSet<object>>()): boolean {
    // 1. Strict equality check (handles primitives, same object references, null, undefined)
    if (a === b) {
        return true;
    }

    // 2. If a or b is null or not an object, and they weren't ===, they are not equal.
    //    (This covers cases like comparing a primitive to an object, or one null and one object)
    if (a == null || typeof a !== 'object' || b == null || typeof b !== 'object') {
        return false;
    }

    // 3. Handle circular references:
    //    If we've already started comparing 'a' and 'b' in the current path,
    //    assume they are equal to break the cycle.
    if (cache.has(a) && cache.get(a)!.has(b)) {
        return true;
    }
    // Add to cache before recursive calls
    if (!cache.has(a)) {
        cache.set(a, new WeakSet());
    }
    cache.get(a)!.add(b);

    // Symmetrically for b -> a to catch cycles like a.c = b; b.c = a;
    if (!cache.has(b)) {
        cache.set(b, new WeakSet());
    }
    cache.get(b)!.add(a);


    // 4. Handle Dates
    if (a instanceof Date && b instanceof Date) {
        return a.getTime() === b.getTime();
    }

    // 5. Handle Regular Expressions
    if (a instanceof RegExp && b instanceof RegExp) {
        return a.source === b.source && a.flags === b.flags;
    }

    // 6. Handle Arrays
    if (Array.isArray(a) && Array.isArray(b)) {
        if (a.length !== b.length) {
            return false;
        }
        for (let i = 0; i < a.length; i++) {
            if (!deepEquals(a[i], b[i], cache)) {
                return false;
            }
        }
        return true;
    }

    // 7. Handle plain Objects
    //    At this point, 'a' and 'b' are objects but not arrays, dates, or regexps.
    //    If one is an array and the other an object, they are not equal.
    if (Array.isArray(a) || Array.isArray(b)) {
        return false;
    }

    const keysA = Object.keys(a);
    const keysB = Object.keys(b);

    if (keysA.length !== keysB.length) {
        return false;
    }

    for (const key of keysA) {
        // Check if key exists in B. `Object.keys` only returns own enumerable properties.
        // `hasOwnProperty` is a more robust check if b might have non-enumerable properties from its prototype with the same name.
        // However, since we compare keysA.length and keysB.length, if key is in keysA but not in keysB,
        // lengths would mismatch or the loop over keysB (if we did that) would find a key in B not in A.
        // The most crucial part is `Object.prototype.hasOwnProperty.call(b, key)`
        if (!Object.prototype.hasOwnProperty.call(b, key)) {
            return false;
        }
        if (!deepEquals(a[key], b[key], cache)) {
            return false;
        }
    }

    return true;
}
