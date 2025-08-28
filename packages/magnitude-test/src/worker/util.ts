import { RegisteredTest, TestOptions } from "@/discovery/types";
import { TestResult, TestState } from "@/runner/state";
import { BrowserOptions, GroundingClient, LLMClient } from "magnitude-core";
import { parentPort, workerData } from "node:worker_threads";
import { isBun } from 'std-env';
import EventEmitter from "node:events";
import { TestFunction } from "@/discovery/types";

declare global {
    var __magnitudeTestFunctions: Map<string, TestFunction> | undefined;
    var __magnitudeMessageEmitter: EventEmitter | undefined;
    var __magnitudeTestHooks: TestHooks | undefined;
    var __magnitudeTestPromptStack: Record<string, string[]> | undefined;
    var __magnitudeTestRegistry: Map<string, TestMetadata> | undefined;
}

if (!globalThis.__magnitudeTestFunctions) {
    globalThis.__magnitudeTestFunctions = new Map<string, TestFunction>();
}
export const testFunctions = globalThis.__magnitudeTestFunctions;

if (!globalThis.__magnitudeMessageEmitter) {
    globalThis.__magnitudeMessageEmitter = new EventEmitter();
}
export const messageEmitter = globalThis.__magnitudeMessageEmitter;


export type TestHooks = Record<
    'beforeAll' | 'afterAll' | 'beforeEach' | 'afterEach',
    (() => void | Promise<void>)[]
>;

export type TestMetadata = {
    title: string;
    url: string;
    filepath: string;
    group?: string;
};

if (!globalThis.__magnitudeTestHooks) {
    globalThis.__magnitudeTestHooks = {
        beforeAll: [],
        afterAll: [],
        beforeEach: [],
        afterEach: [],
    };
}
export const hooks = globalThis.__magnitudeTestHooks;

if (!globalThis.__magnitudeTestPromptStack) {
    globalThis.__magnitudeTestPromptStack = {};
}
export const testPromptStack = globalThis.__magnitudeTestPromptStack;

if (!globalThis.__magnitudeTestRegistry) {
    globalThis.__magnitudeTestRegistry = new Map<string, TestMetadata>();
}
export const testRegistry = globalThis.__magnitudeTestRegistry;

export type TestWorkerIncomingMessage = {
    type: "execute"
    testId: string;
} | {
    type: "graceful_shutdown"
}

export type TestWorkerOutgoingMessage = {
    type: "load_complete";
} | {
    type: "load_error";
    error: string;
} | {
    type: "registered";
    test: RegisteredTest;
} | {
    type: "test_result";
    testId: string;
    result: TestResult;
} | {
    type: "test_error";
    testId: string;
    error: string;
} | {
    type: "test_state_change";
    testId: string;
    state: TestState;
} | {
    type: "graceful_shutdown_complete";
}

export function postToParent(message: TestWorkerOutgoingMessage) {
    if (isBun) {
        if (typeof process.send !== 'function') {
            throw new Error("Not running in a Bun subprocess with IPC");
        }
        process.send(message);
        return;
    }
    if (!parentPort) throw new Error("Not running in a worker thread");
    parentPort.postMessage(message);
}

export type TestWorkerData = {
    absoluteFilePath: string;
    options: TestOptions;
    relativeFilePath: string;
    browserOptions?: BrowserOptions;
    llm?: LLMClient | LLMClient[];
    grounding?: GroundingClient;
    telemetry?: boolean;
}

export function getTestWorkerData() {
    if (isBun) {
        const dataStr = process.env.MAGNITUDE_WORKER_DATA;
        if (!dataStr) {
            throw new Error('Worker data not found in environment');
        }
        return JSON.parse(dataStr) as TestWorkerData;
    }
    if (!parentPort) {
        throw new Error('Do not use this module on the main thread');
    }
    return workerData as TestWorkerData;
}
