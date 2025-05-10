import { TestDeclaration, TestOptions, TestFunction, TestGroupFunction, TestRunnable } from './types';
import { TestRegistry } from './testRegistry';
import { addProtocolIfMissing } from '@/util';


function testDecl(
    title: string,
    optionsOrTestFn: TestOptions | TestFunction,
    testFnOrNothing?: TestFunction
): void {
    // First deal with the weird parameter ordering
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

    // Get global registry
    const registry = TestRegistry.getInstance();
    const combinedOptions = { ...registry.getActiveOptions(), ...(options ?? {}) };

    if (!combinedOptions.url) {
        throw Error("URL must be provided either through (1) env var MAGNITUDE_TEST_URL, (2) via test.config, or (3) in group or test options");
    }

    const baseUrl = addProtocolIfMissing(combinedOptions.url);
    
    let finalUrl = baseUrl;
    if (combinedOptions.path) {
        try {
            finalUrl = new URL(combinedOptions.path, baseUrl).toString();
        } catch (error) {
            throw new Error(`Failed to combine URL '${baseUrl}' with path '${combinedOptions.path}': ${error}`);
        }
    }

    // Add the declared test function as a runnable to the registry
    const runnable: TestRunnable = {
        fn: testFn,
        title: title,
        url: finalUrl
    }
    registry.register(runnable);

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

    const registry = TestRegistry.getInstance();

    // Set active group context
    registry.setCurrentGroup({
        name: id,
        options: options
    });

    // Run the block to register the test cases with the group context
    testFn();

    // Remove active group context
    registry.unsetCurrentGroup();
}

// testDecl.config = function (
//     options: TestGlobalConfig
// ) {
//     const registry = TestRegistry.getInstance();
//     registry.setGlobalOptions(options);
// }

export const test = testDecl as TestDeclaration;
