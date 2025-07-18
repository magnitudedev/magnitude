import { TestFunction, TestGroup, TestOptions } from "@/discovery/types";
import cuid2 from "@paralleldrive/cuid2";
import { getTestWorkerData, postToParent, testFunctions, messageEmitter, TestWorkerIncomingMessage } from "./util";
import { TestCaseAgent } from "@/agent";
import { TestResult, TestState, TestStateTracker } from "@/runner/state";
import { buildDefaultBrowserAgentOptions } from "magnitude-core";
import { sendTelemetry } from "@/runner/telemetry";
import { testPromptStack } from "@/worker/testDeclaration";

// This module has to be separate so it only gets imported once after possible compilation by jiti.

const workerData = getTestWorkerData();

const generateId = cuid2.init({ length: 12 });

export function registerTest(testFn: TestFunction, title: string, url: string) {
    const testId = generateId();
    testFunctions.set(testId, testFn);
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

let currentGroup: TestGroup | undefined;
export function setCurrentGroup(group?: TestGroup) {
    currentGroup = group;
}
export function currentGroupOptions(): TestOptions {
    return structuredClone(currentGroup?.options) ?? {};
}

messageEmitter.removeAllListeners('message');
messageEmitter.on('message', async (message: TestWorkerIncomingMessage) => {
    if (message.type !== 'execute') return;
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
            await testFn(agent);
            finalState = {
                ...tracker.getState(),
                status: 'passed' as const,
                doneAt: Date.now()
            };

            finalResult = { passed: true };
        } catch (error) {
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
            result: { ...finalResult, state: finalState }
        });
    } catch (error) {
        postToParent({
            type: 'test_error',
            error: error instanceof Error ? error.message : String(error),
            testId: test.id,
        });
    }
});
