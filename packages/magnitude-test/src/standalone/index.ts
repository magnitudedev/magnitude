import logger from '@/logger';
import { TestCaseAgent, Magnus, AgentStateTracker, StepDescriptor } from 'magnitude-core';
import { BrowserContext, chromium, LaunchOptions, Page } from 'playwright';
import { tryDeriveEnvironmentPlannerClient, sendTelemetry } from '@/util';
import { processUrl } from '@/discovery/testRegistry';
import { MagnitudeConfig, TestFunctionContext } from '@/discovery/types';

/**
 * Initialize Magnitude and return wrapper functions for arbitrary test runners to use
 * 
 * @param config Configuration options for Magnitude
 * @returns Object containing withMagnitude function
 */
export async function createMagnitude(config: MagnitudeConfig) {
    const { signal } = new AbortController();

    const planner = config.planner || tryDeriveEnvironmentPlannerClient();
    if (!planner) {
        throw new Error("No planner client configured. Set an appropriate environment variable or provide a planner in the config.");
    }

    let executor = config.executor;
    if (!executor) {
        const apiKey = process.env.MOONDREAM_API_KEY;
        if (!apiKey) {
            throw new Error("Missing MOONDREAM_API_KEY, get one at https://moondream.ai/c/cloud/api-keys");
        }
        executor = {
            provider: 'moondream',
            options: {
                apiKey
            }
        };
    }

    const defaultOptions: LaunchOptions = {
        headless: false,
        args: ['--disable-gpu']
    };

    const launchOptions = {
        ...defaultOptions,
        ...config.browser?.launchOptions
    };

    // Create the browser instance
    const browser = await chromium.launch(launchOptions);

    return {
        signal,
        withMagnitude: <T extends unknown[], R>(url: string, testFn: (context: TestFunctionContext, ...rest: T) => R) =>
            (async (...rest: T) => {
                const startingUrl = processUrl(config.url, url) ?? url;

                const agent = new TestCaseAgent({
                    planner,
                    executor,
                    browserContextOptions: { ...(config.browser?.contextOptions ?? {}), baseURL: startingUrl },
                    signal
                });

                const stateTracker = new AgentStateTracker(agent);

                try {
                    await agent.start(browser, startingUrl);

                    const r = await testFn({
                        ai: new Magnus(agent),
                        get page(): Page {
                            return agent.getPage();
                        },
                        get context(): BrowserContext {
                            return agent.getContext();
                        }
                    }, ...rest);

                    return r;
                } finally {
                    try {
                        await agent.close();
                    } catch (closeErr: unknown) {
                        logger.warn(`Error during agent.close: ${closeErr}`);
                    }
                    const state = stateTracker.getState();
                    const numSteps = state.stepsAndChecks.filter(item => item.variant === 'step').length;
                    const numChecks = state.stepsAndChecks.filter(item => item.variant === 'check').length;
                    const actionCount = state.stepsAndChecks
                        .filter((item): item is StepDescriptor => item.variant === 'step')
                        .reduce((sum, step) => sum + step.actions.length, 0);

                    // Send telemetry if enabled (default to true if not specified)
                    const telemetryEnabled = config.telemetry !== false;
                    if (telemetryEnabled) {
                        await sendTelemetry({
                            startedAt: state.startedAt ?? Date.now(),
                            doneAt: Date.now(),
                            macroUsage: state.macroUsage,
                            microUsage: state.microUsage,
                            cached: state.cached ?? false,
                            testCase: {
                                numChecks: numChecks,
                                numSteps: numSteps,
                            },
                            actionCount: actionCount,
                            result: state.failure?.variant ?? 'passed'
                        });
                    }
                }
            }) as (...rest: T) => Promise<Awaited<R>> // https://github.com/microsoft/TypeScript/issues/56083
    };
}
