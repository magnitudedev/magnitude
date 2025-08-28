import { TestFunction, TestGroup, TestOptions } from "@/discovery/types";
import cuid2 from "@paralleldrive/cuid2";
import { getTestWorkerData, postToParent, testFunctions, messageEmitter, TestWorkerIncomingMessage, hooks, testRegistry, testPromptStack } from "./util";
import { TestCaseAgent } from "@/agent";
import { TestResult, TestState, TestStateTracker } from "@/runner/state";
import { buildDefaultBrowserAgentOptions } from "magnitude-core";
import { sendTelemetry } from "@/runner/telemetry";

// This module has to be separate so it only gets imported once after possible compilation by jiti.

const workerData = getTestWorkerData();

const generateId = cuid2.init({ length: 12 });

export function registerTest(testFn: TestFunction, title: string, url: string) {
    const testId = generateId();
    testFunctions.set(testId, testFn);

    testRegistry.set(testId, {
        title,
        url,
        filepath: workerData.relativeFilePath,
        group: currentGroup?.name
    });

    postToParent({
        type: 'registered',
        test: {
            id: testId,
            title,
            url,
            filepath: workerData.relativeFilePath,
            group: currentGroup?.name
        }
    });
}

let beforeAllExecuted = false;
let beforeAllError: Error | null = null;
let afterAllExecuted = false;
let isShuttingDown = false;
let pendingAfterEach: Set<string> = new Set();
// No state reset is needed because each test file is run in a separate worker

let currentGroup: TestGroup | undefined;
export function setCurrentGroup(group?: TestGroup) {
    currentGroup = group;
}
export function currentGroupOptions(): TestOptions {
    return structuredClone(currentGroup?.options) ?? {};
}

messageEmitter.removeAllListeners('message');
messageEmitter.on('message', async (message: TestWorkerIncomingMessage) => {
    if (message.type === 'graceful_shutdown') {
        isShuttingDown = true;

        if (pendingAfterEach.size > 0) {
            try {
                await Promise.all(
                    [...pendingAfterEach].map(async (_testId) => {
                        for (const afterEachHook of hooks.afterEach) {
                            await afterEachHook();
                        }
                    })
                );
            } catch (error) {
                console.error("afterEach hooks failed during graceful shutdown:", error);
            }
        }

        if (!afterAllExecuted) {
            try {
                for (const afterAllHook of hooks.afterAll) {
                    await afterAllHook();
                }
                afterAllExecuted = true;
            } catch (error) {
                console.error("afterAll hook failed during graceful shutdown:\n", error);
            }
        }

        postToParent({ type: 'graceful_shutdown_complete' });
        return;
    }


    if (message.type !== 'execute') return;

    // Don't start new tests if shutting down
    if (isShuttingDown) {
        postToParent({
            type: 'test_error',
            testId: message.testId,
            error: 'Test cancelled due to graceful shutdown'
        });
        return;
    }

    const { testId } = message;
    const { browserOptions, llm, grounding, telemetry } = workerData;
    const testFn = testFunctions.get(testId);

    if (!testFn) {
        postToParent({
            type: 'test_error',
            testId: testId,
            error: `Test function not found: ${testId}`
        });
        return;
    }

    const testMetadata = testRegistry.get(testId);
    if (!testMetadata) {
        postToParent({
            type: 'test_error',
            testId: testId,
            error: `Test metadata not found: ${testId}`
        });
        return;
    }

    try {
        const promptStack = testPromptStack[testMetadata.title] || [];
        const prompt = promptStack.length > 0 ? promptStack.join('\n') : undefined;
        const { agentOptions: defaultAgentOptions, browserOptions: defaultBrowserOptions } = buildDefaultBrowserAgentOptions({
            agentOptions: { llm, ...(prompt ? { prompt } : {}) },
            browserOptions: {
                url: testMetadata.url,
                browser: browserOptions,
                grounding
            }
        });

        const agent = new TestCaseAgent({
            // disable telemetry to keep test run telemetry seperate from general automation telemetry
            agentOptions: { ...defaultAgentOptions, telemetry: false },
            browserOptions: defaultBrowserOptions,
        });

        const tracker = new TestStateTracker(agent);

        tracker.events.on('stateChanged', (state) => {
            postToParent({
                type: 'test_state_change',
                testId: testId,
                state
            });
        });

        await agent.start();

        let finalState: TestState;
        let finalResult: TestResult;

        try {
            if (!beforeAllExecuted && hooks.beforeAll.length > 0) {
                try {
                    for (const beforeAllHook of hooks.beforeAll) {
                        await beforeAllHook();
                    }
                } catch (error) {
                    console.error("beforeAll hooks failed:", error);
                    beforeAllError = error instanceof Error ? error : new Error(String(error));
                } finally {
                    beforeAllExecuted = true;
                }
            }

            if (beforeAllError) {
                throw new Error(`beforeAll hook failed: ${beforeAllError.message}`);
            }

            for (const beforeEachHook of hooks.beforeEach) {
                try {
                    await beforeEachHook();
                } catch (error) {
                    console.error(`beforeEach hook failed for test '${testMetadata.title}':`, error);
                    throw error;
                }
            }
            pendingAfterEach.add(testId);

            await testFn(agent);

            if (!isShuttingDown) {
                pendingAfterEach.delete(testId);
                for (const afterEachHook of hooks.afterEach) {
                    try {
                        await afterEachHook();
                    } catch (error) {
                        console.error(`afterEach hook failed for test '${testMetadata.title}':`, error);
                        throw error;
                    }
                }
            }

            finalState = {
                ...tracker.getState(),
                status: 'passed' as const,
                doneAt: Date.now()
            };

            finalResult = { passed: true };
        } catch (error) {
            if (!isShuttingDown) {
                pendingAfterEach.delete(testId);
                try {
                    for (const afterEachHook of hooks.afterEach) {
                        await afterEachHook();
                    }
                } catch (afterEachError) {
                    console.error(`afterEach hook failed for failing test '${testMetadata.title}':`, afterEachError);
                    const originalMessage = error instanceof Error ? error.message : String(error);
                    const afterEachMessage = afterEachError instanceof Error ? afterEachError.message : String(afterEachError);
                    error = new Error(`Test failed: ${originalMessage}. Additionally, afterEach hook failed: ${afterEachMessage}`);
                }
            }

            const failure = {
                message: error instanceof Error ? error.message : String(error)
            };

            finalState = {
                ...tracker.getState(),
                failure: failure,
                status: 'failed' as const,
                doneAt: Date.now()
            };

            finalResult = { passed: false, failure };
        }

        await agent.stop();

        postToParent({
            type: 'test_state_change',
            testId: testId,
            state: finalState
        });

        if (finalState && (telemetry ?? true)) await sendTelemetry(finalState);

        postToParent({
            type: 'test_result',
            testId: testId,
            result: finalResult ??
                { passed: false, failure: { message: "Test result doesn't exist" } },
        });
    } catch (error) {
        postToParent({
            type: 'test_error',
            error: error instanceof Error ? error.message : String(error),
            testId: testId,
        });
    }
});
