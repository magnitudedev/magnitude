import { TestFunction, TestGroup, TestOptions, RegisteredTest } from "@/discovery/types";
import cuid2 from "@paralleldrive/cuid2";
import { getTestWorkerData, postToParent, testFunctions, messageEmitter, TestWorkerIncomingMessage, hooks, testRegistry, testPromptStack, getOrInitGroupHookSet } from "./util";
import { TestCaseAgent } from "@/agent";
import { TestResult, TestState, TestStateTracker } from "@/runner/state";
import { buildDefaultBrowserAgentOptions } from "magnitude-core";
import { sendTelemetry } from "@/runner/telemetry";

// This module has to be separate so it only gets imported once after possible compilation by jiti.

const workerData = getTestWorkerData();

const generateId = cuid2.init({ length: 12 });

/** Get all prefix keys for a hierarchy in outer → inner order */
function keysForPrefixes(hierarchy: TestGroup[]): string[] {
    const keys: string[] = [];
    for (let i = 1; i <= hierarchy.length; i++) {
        keys.push(hierarchy.slice(0, i).map(g => g.id).join('>'));
    }
    return keys;
}

export function registerTest(testFn: TestFunction, title: string, url: string) {
    const testId = generateId();
    testFunctions.set(testId, testFn);
    const groupHierarchy = getCurrentGroupHierarchy().map(({ name, id }) => ({ name, id }));

    testRegistry.set(testId, {
        title,
        url,
        filepath: workerData.relativeFilePath,
        group: getCurrentGroup()?.name,
        groupHierarchy
    });
    postToParent({
        type: 'registered',
        test: {
            id: testId,
            title,
            url,
            filepath: workerData.relativeFilePath,
            group: getCurrentGroup()?.name,
            groupHierarchy
        }
    });
}

let beforeAllExecuted = false;
let beforeAllError: Error | null = null;
let afterAllExecuted = false;
let isShuttingDown = false;
let pendingAfterEach: Map<string, RegisteredTest> = new Map();
let groupBeforeAllExecuted: Set<string> = new Set();
let groupBeforeAllErrors: Map<string, Error> = new Map();
let executedBeforeAllOrder: string[] = [];
// No state reset is needed because each test file is run in a separate worker

let currentGroupStack: TestGroup[] = [];
export function pushCurrentGroup(group: TestGroup) {
    currentGroupStack.push(group);
}
export function popCurrentGroup() {
    currentGroupStack.pop();
}
export function getCurrentGroup(): TestGroup | undefined {
    return currentGroupStack.length > 0 ? currentGroupStack[currentGroupStack.length - 1] : undefined;
}
export function getCurrentGroupHierarchy(): TestGroup[] {
    return [...currentGroupStack];
}
export function currentGroupOptions(): TestOptions {
    let mergedOptions: TestOptions = {};
    for (const group of currentGroupStack) {
        if (group.options) {
            mergedOptions = { ...mergedOptions, ...group.options };
        }
    }
    return structuredClone(mergedOptions);
}

async function executeAfterEachHooks(test: RegisteredTest) {
    // Run group-level afterEach hooks in inner → outer order
    if (test.groupHierarchy && test.groupHierarchy.length > 0) {
        const prefixKeys = keysForPrefixes(test.groupHierarchy);
        for (const key of prefixKeys.reverse()) {
            const hookSet = getOrInitGroupHookSet(key);
            for (const afterEachHook of hookSet.afterEach) {
                try {
                    await afterEachHook();
                } catch (error) {
                    const groupNames = test.groupHierarchy.map(g => g.name).join(' > ');
                    console.error(`Group afterEach hook failed for test '${test.title}' in hierarchy '${groupNames}':`, error);
                    throw error;
                }
            }
        }
    }

    for (const afterEachHook of hooks.afterEach) {
        try {
            await afterEachHook();
        } catch (error) {
            console.error(`afterEach hook failed for test '${test.title}':`, error);
            throw error;
        }
    }
}

async function executeGroupBeforeAllHooks(test: RegisteredTest) {
    if (!test.groupHierarchy || test.groupHierarchy.length === 0) {
        return;
    }

    const prefixKeys = keysForPrefixes(test.groupHierarchy);

    for (const key of prefixKeys) {
        if (groupBeforeAllExecuted.has(key)) {
            const error = groupBeforeAllErrors.get(key);
            if (error) {
                const groupNames = test.groupHierarchy.map(g => g.name).join(' > ');
                throw new Error(`Group beforeAll hook failed for hierarchy '${groupNames}': ${error.message}`);
            }
            continue;
        }

        groupBeforeAllExecuted.add(key);
        executedBeforeAllOrder.push(key);

        try {
            const hookSet = getOrInitGroupHookSet(key);
            for (const beforeAllHook of hookSet.beforeAll) {
                await beforeAllHook();
            }
        } catch (error) {
            const hookError = error instanceof Error ? error : new Error(String(error));
            const groupNames = test.groupHierarchy.map(g => g.name).join(' > ');
            console.error(`Group beforeAll hook failed for hierarchy '${groupNames}':`, hookError);
            groupBeforeAllErrors.set(key, hookError);
            throw hookError;
        }
    }
}

messageEmitter.removeAllListeners('message');
messageEmitter.on('message', async (message: TestWorkerIncomingMessage) => {
    if (message.type === 'graceful_shutdown') {
        isShuttingDown = true;

        try {
            if (pendingAfterEach.size > 0) {
                try {
                    await Promise.all(
                        [...pendingAfterEach.values()].map(async (test) => {
                            try {
                                await executeAfterEachHooks(test);
                            } catch (error) {
                                console.error(`afterEach hooks failed during graceful shutdown for test '${test.title}':`, error);
                                // Don't throw here - we want to continue with other tests
                            }
                        })
                    );
                } catch (error) {
                    console.error("afterEach hooks failed during graceful shutdown:", error);
                }
            }

            if (!afterAllExecuted) {
                try {
                    // Run group afterAll hooks in reverse execution order (inner → outer)
                    for (const key of executedBeforeAllOrder.reverse()) {
                        const hookSet = getOrInitGroupHookSet(key);
                        if (hookSet.afterAll.length > 0) {
                            try {
                                for (const afterAllHook of hookSet.afterAll) {
                                    await afterAllHook();
                                }
                            } catch (error) {
                                console.error(`Group afterAll hook failed during graceful shutdown for hierarchy key '${key}':`, error);
                            }
                        }
                    }

                    for (const afterAllHook of hooks.afterAll) {
                        await afterAllHook();
                    }
                    afterAllExecuted = true;
                } catch (error) {
                    console.error("afterAll hook failed during graceful shutdown:\n", error);
                }
            }

            postToParent({ type: 'graceful_shutdown_complete' });
        } catch (error) {
            console.error("Critical error during graceful shutdown:", error);
            postToParent({ type: 'graceful_shutdown_complete' });
        }
        return;
    }


    if (message.type !== 'execute') return;

    // Don't start new tests if shutting down
    if (isShuttingDown) {
        postToParent({
            type: 'test_error',
            testId: message.test.id,
            error: 'Test cancelled due to graceful shutdown'
        });
        return;
    }

    const { test, browserOptions, llm, grounding, telemetry } = message;
    const testFn = testFunctions.get(test.id);

    if (!testFn) {
        postToParent({
            type: 'test_error',
            testId: test.id,
            error: `Test function not found: ${test.id}`
        });
        return;
    }

    try {
        const promptStack = testPromptStack[test.title] || [];
        const prompt = promptStack.length > 0 ? promptStack.join('\n') : undefined;
        const { agentOptions: defaultAgentOptions, browserOptions: defaultBrowserOptions } = buildDefaultBrowserAgentOptions({
            agentOptions: { llm, ...(prompt ? { prompt } : {}) },
            browserOptions: {
                url: test.url,
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
                testId: test.id,
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

            await executeGroupBeforeAllHooks(test);

            for (const beforeEachHook of hooks.beforeEach) {
                try {
                    await beforeEachHook();
                } catch (error) {
                    console.error(`beforeEach hook failed for test '${test.title}':`, error);
                    throw error;
                }
            }

            // Run group-level beforeEach hooks in outer → inner order
            if (test.groupHierarchy && test.groupHierarchy.length > 0) {
                const prefixKeys = keysForPrefixes(test.groupHierarchy);
                for (const key of prefixKeys) {
                    const hookSet = getOrInitGroupHookSet(key);
                    for (const beforeEachHook of hookSet.beforeEach) {
                        try {
                            await beforeEachHook();
                        } catch (error) {
                            const groupNames = test.groupHierarchy.map(g => g.name).join(' > ');
                            console.error(`Group beforeEach hook failed for test '${test.title}' in hierarchy '${groupNames}':`, error);
                            throw error;
                        }
                    }
                }
            }
            pendingAfterEach.set(test.id, test);

            await testFn(agent);

            if (!isShuttingDown) {
                pendingAfterEach.delete(test.id);
                await executeAfterEachHooks(test);
            }

            finalState = {
                ...tracker.getState(),
                status: 'passed' as const,
                doneAt: Date.now()
            };

            finalResult = { passed: true };
        } catch (error) {
            if (!isShuttingDown) {
                pendingAfterEach.delete(test.id);
                try {
                    await executeAfterEachHooks(test);
                } catch (afterEachError) {
                    console.error(`afterEach hook failed for failing test '${test.title}':`, afterEachError);
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
            testId: test.id,
            state: finalState
        });

        if (finalState && (telemetry ?? true)) await sendTelemetry(finalState);

        postToParent({
            type: 'test_result',
            testId: test.id,
            result: finalResult ??
                { passed: false, failure: { message: "Test result doesn't exist" } },
        });
    } catch (error) {
        postToParent({
            type: 'test_error',
            error: error instanceof Error ? error.message : String(error),
            testId: test.id,
        });
    }
});
