import { TestDeclaration, TestOptions, TestFunction, TestGroupFunction } from '../discovery/types';
import { addProtocolIfMissing, processUrl } from '@/util';
import { getTestWorkerData, hooks, TestHooks, testPromptStack } from '@/worker/util';
import { currentGroupOptions, registerTest, setCurrentGroup } from '@/worker/localTestRegistry';

const workerData = getTestWorkerData();

function testDecl(
    title: string,
    optionsOrTestFn: TestOptions | TestFunction,
    testFnOrNothing?: TestFunction
): void {
    let options: TestOptions;
    let testFn: TestFunction;

    if (typeof optionsOrTestFn == 'function') {
        options = {};
        testFn = optionsOrTestFn
    }
    else {
        options = optionsOrTestFn;
        if (!testFnOrNothing) {
            throw new Error("Test function is required");
        }
        testFn = testFnOrNothing;
    }

    const groupOptions = currentGroupOptions();

    const combinedOptions: TestOptions = {
        ...(workerData.options ?? {}),
        ...groupOptions,
        ...(options ?? {}),
        url: processUrl(workerData.options?.url, groupOptions.url, options?.url)
    };

    if (!combinedOptions.url) {
        throw Error("URL must be provided either through (1) env var MAGNITUDE_TEST_URL, (2) via magnitude.config.ts, or (3) in group or test options");
    }

    // Stack group and test prompts (group first, then test)
    const promptStack: string[] = [];
    if (groupOptions.prompt) promptStack.push(groupOptions.prompt);
    if (options.prompt) promptStack.push(options.prompt);
    testPromptStack[title] = promptStack;

    registerTest(testFn, title, addProtocolIfMissing(combinedOptions.url));

    // TODO: maybe return an object to enable some kind of chaining
}

testDecl.group = function (
    id: string,
    optionsOrTestFn: TestOptions | TestGroupFunction,
    testFnOrNothing?: TestGroupFunction
): void {
    let options: TestOptions;
    let testFn: TestGroupFunction;

    if (typeof optionsOrTestFn == 'function') {
        options = {};
        testFn = optionsOrTestFn
    }
    else {
        options = optionsOrTestFn;
        if (!testFnOrNothing) {
            throw new Error("Test function is required");
        }
        testFn = testFnOrNothing;
    }

    setCurrentGroup({ name: id, options });
    testFn();
    setCurrentGroup(undefined);
}

export const test = testDecl as TestDeclaration;

function createHookRegistrar(kind: keyof TestHooks) {
    return function (fn: TestHooks[typeof kind][number]) {
        if (typeof fn !== "function") {
            throw new Error(`${kind} expects a function`);
        }
        hooks[kind].push(fn);
    };
}

export const beforeAll = createHookRegistrar("beforeAll");
export const afterAll = createHookRegistrar("afterAll");
export const beforeEach = createHookRegistrar("beforeEach");
export const afterEach = createHookRegistrar("afterEach");
