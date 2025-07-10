import {
  TestDeclaration,
  TestOptions,
  TestFunction,
  TestGroupFunction,
} from "./types";
import { TestRegistry } from "./testRegistry";
import { addProtocolIfMissing, processUrl } from "@/util";

function testDecl(
  title: string,
  optionsOrTestFn: TestOptions | TestFunction,
  testFnOrNothing?: TestFunction
): void {
  // First deal with the weird parameter ordering
  let options: TestOptions;
  let testFn: TestFunction;

  if (typeof optionsOrTestFn == "function") {
    options = {};
    testFn = optionsOrTestFn;
  } else {
    options = optionsOrTestFn;
    if (!testFnOrNothing) {
      throw new Error("Test function is required");
    }
    testFn = testFnOrNothing;
  }

  // Get global registry
  const registry = TestRegistry.getInstance();
  const registryOptions = registry.getActiveOptions();
  const combinedOptions = {
    ...registryOptions,
    ...(options ?? {}),
    url: processUrl(registryOptions.url, options?.url),
  };

  if (!combinedOptions.url) {
    throw Error(
      "URL must be provided either through (1) env var MAGNITUDE_TEST_URL, (2) via magnitude.config.ts, or (3) in group or test options"
    );
  }

  // Add the declared test function as a runnable to the registry
  registry.register({
    fn: testFn,
    title: title,
    url: addProtocolIfMissing(combinedOptions.url),
  });

  // TODO: maybe return an object to enable some kind of chaining
}

testDecl.group = function (
  id: string,
  optionsOrTestFn: TestOptions | TestGroupFunction,
  testFnOrNothing?: TestGroupFunction
): void {
  let options: TestOptions;
  let testFn: TestGroupFunction;

  if (typeof optionsOrTestFn == "function") {
    options = {};
    testFn = optionsOrTestFn;
  } else {
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
    options: options,
  });

  // Run the block to register the test cases with the group context
  testFn();

  // Remove active group context
  registry.unsetCurrentGroup();
};

// testDecl.config = function (
//     options: TestGlobalConfig
// ) {
//     const registry = TestRegistry.getInstance();
//     registry.setGlobalOptions(options);
// }

export const test = testDecl as TestDeclaration;

// ────────────────────────────────────────────────
//  Hook DSL (beforeAll / afterAll / beforeEach / afterEach)
// ────────────────────────────────────────────────

import type { HookFn } from "./testRegistry";

function createHookRegistrar(kind: keyof import("./testRegistry").TestHooks) {
  return function (fn: HookFn) {
    if (typeof fn !== "function") {
      throw new Error(`${kind} expects a function`);
    }
    const registry = TestRegistry.getInstance();
    registry.addHook(kind as any, fn);
  };
}

export const beforeAll = createHookRegistrar("beforeAll");
export const afterAll = createHookRegistrar("afterAll");
export const beforeEach = createHookRegistrar("beforeEach");
export const afterEach = createHookRegistrar("afterEach");
