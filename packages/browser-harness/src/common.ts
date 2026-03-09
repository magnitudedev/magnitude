export type RetryMode = {
    mode: 'retry_on_partial_message',
    errorSubstrings: string[],
} | {
    mode: 'retry_all',
};

export type RetryParams = {
    retryLimit: number,
    delayMs: number,
    showWarnOnRetry: boolean,
}

export type RetryOptions = RetryMode & RetryParams;

export async function retryOnErrorIsSuccess<T>(
    fnToRetry: () => Promise<T>,
    retryOptions: RetryMode & Partial<RetryParams>
): Promise<boolean> {
    try {
        await retryOnError(fnToRetry, retryOptions);
        return true;
    } catch (error) {
        return false;
    }
}

export async function retryOnError<T>(
    fnToRetry: () => Promise<T>,
    retryOptions: RetryMode & Partial<RetryParams>
): Promise<T> {
    let lastError: any;

    const options: RetryOptions = {
        ...retryOptions,
        retryLimit: retryOptions.retryLimit ?? 3,
        delayMs: retryOptions.delayMs ?? 200,
        showWarnOnRetry: retryOptions.showWarnOnRetry ?? false,
    } as RetryOptions;

    if (options.retryLimit < 0) {
        options.retryLimit = 0;
    }

    for (let attempt = 0; attempt <= options.retryLimit; attempt++) {
        try {
            return await fnToRetry();
        } catch (error: any) {
            lastError = error;

            const errorMessage = String(error?.message ?? error);

            if (options.mode === 'retry_all') {
                if (options.showWarnOnRetry) {
                    console.warn(`Retrying on: ${errorMessage}`);
                }
            } else if (options.mode === 'retry_on_partial_message') {
                const includesSubstring = options.errorSubstrings.some((substring) => errorMessage.includes(substring));

                if (includesSubstring) {
                    if (options.showWarnOnRetry) {
                        console.warn(`Retrying on: ${errorMessage}`);
                    }
                } else {
                    // Error message does NOT contain the target substring. This error is not retryable.
                    throw lastError;
                }
            }
        }
        await new Promise((resolve) => setTimeout(resolve, options.delayMs));
    }

    throw lastError;
}
