import {
  TestOptions,
  TestGroup,
  MagnitudeConfig,
  TestFunction,
  RegisteredTest,
} from "./types";
import { TestCompiler } from "@/compiler";
import { processUrl } from "@/util";
import cuid2 from "@paralleldrive/cuid2";
import { pathToFileURL } from "node:url";

// === Hook system types ===
export type HookFn = () => void | Promise<void>;

export interface TestHooks {
  beforeAll: HookFn[];
  afterAll: HookFn[];
  beforeEach: HookFn[];
  afterEach: HookFn[];
}

const generateId = cuid2.init({ length: 12 });

// Warning: The registry is bundled with every test, be careful about the module tree here.

declare global {
  var __testRegistry: TestRegistry | undefined;
}

export class TestRegistry {
  //private static instance: TestRegistry;

  // map from filepath to ungrouped & grouped test cases
  private tests: RegisteredTest[] = [];
  //private tests: CategorizedTestCases = {};

  private currentGroup?: TestGroup;
  private currentFilePath?: string;

  // Add global hooks storage
  private hooks: TestHooks = {
    beforeAll: [],
    afterAll: [],
    beforeEach: [],
    afterEach: [],
  };

  private globalOptions!: MagnitudeConfig;

  private compiler: TestCompiler;

  private constructor() {
    this.compiler = new TestCompiler();
  }

  public static getInstance(): TestRegistry {
    // Use globalThis to ensure same instance is used across module boundaries
    if (!(globalThis as any).__magnitude__) {
      (globalThis as any).__magnitude__ = {};
    }

    if (!(globalThis as any).__magnitude__.registry) {
      (globalThis as any).__magnitude__.registry = new TestRegistry();
    }

    return (globalThis as any).__magnitude__.registry;
  }

  public register(testCase: {
    fn: TestFunction;
    title: string;
    url: string;
  }): void {
    // Register a test case to be tracked by test runner
    if (!this.currentFilePath) {
      throw Error("File path context not set before registering test");
    }

    const id = generateId();
    const filepath = this.currentFilePath;

    this.tests.push({
      id: id,
      fn: testCase.fn,
      title: testCase.title,
      url: testCase.url,
      filepath: filepath,
      group: this.currentGroup?.name,
    });

    // if (!(this.currentFilePath in this.tests)) {
    //     this.tests[this.currentFilePath] = { ungrouped: [], groups: {} }
    // }
    // const testsForPath = this.tests[this.currentFilePath];

    // if (this.currentGroup) {
    //     const groupName = this.currentGroup.name;
    //     if (!(groupName in testsForPath.groups)) {
    //         testsForPath.groups[groupName] = [];
    //     }
    //     testsForPath.groups[groupName].push(testCase);
    // } else {
    //     testsForPath.ungrouped.push(testCase);
    // }
  }

  getRegisteredTests(): RegisteredTest[] {
    return this.tests;
  }

  // public getRegisteredTestCases(): CategorizedTestCases {
  //     return this.tests;
  // }

  // public getFlattenedTestCases(): Array<CategorizedTestRunnable> {
  //     const tests = [];

  //     for (const filePath in this.tests) {
  //         // Add ungrouped tests
  //         for (const runnable of this.tests[filePath].ungrouped) {
  //             tests.push({ ...runnable, file: filePath, group: null });
  //         }

  //         // Add grouped tests
  //         for (const groupName in this.tests[filePath].groups) {
  //             for (const runnable of this.tests[filePath].groups[groupName]) {
  //                 tests.push({ ...runnable, file: filePath, group: groupName });
  //             }
  //         }
  //     }

  //     return tests;
  // }

  public setCurrentGroup(group: TestGroup): void {
    //this.currentGroupName = groupName;
    this.currentGroup = group;
  }

  public unsetCurrentGroup(): void {
    this.currentGroup = undefined;
  }

  public setCurrentFilePath(filePath: string): void {
    //console.log("setCurrentFilePath:", filePath);
    this.currentFilePath = filePath;
    //console.log("currentFilePath:", this.currentFilePath);
  }

  public unsetCurrentFilePath(): void {
    //console.log("unsetCurrentFilePath:", this.currentFilePath);
    this.currentFilePath = undefined;
  }

  public setGlobalOptions(options: MagnitudeConfig): void {
    this.globalOptions = options;
  }

  // Get current option overrides for the active group scope
  public getActiveOptions(): TestOptions {
    const envOptions = process.env.MAGNITUDE_TEST_URL
      ? {
          url: process.env.MAGNITUDE_TEST_URL,
        }
      : {};

    return {
      ...this.globalOptions,
      ...envOptions, // env options take precedence over global options
      ...(this.currentGroup?.options ?? {}),
      url: processUrl(
        envOptions.url,
        this.globalOptions.url,
        this.currentGroup?.options?.url
      ),
    };
  }

  async loadTestFile(
    absoluteFilePath: string,
    relativeFilePath: string
  ): Promise<void> {
    // adding this back in causes ES module err
    //logUpdate("foo")
    try {
      // Set current file path in registry
      this.setCurrentFilePath(relativeFilePath);

      // Compile the file
      const compiledPath = await this.compiler.compileFile(absoluteFilePath);

      // Import the compiled file - triggering it to register its test cases
      await import(pathToFileURL(compiledPath).href);

      //console.log(`Loaded test file: ${relativeFilePath}`);

      // Notify the viewer about the loaded file
      //this.viewer.addLoadedFile(relativeFilePath);
    } catch (error) {
      console.error(`Failed to load test file ${relativeFilePath}:`, error);
      throw error;
    } finally {
      // Always unset the current file path when done
      this.unsetCurrentFilePath();
    }
  }

  public addHook(kind: keyof TestHooks, fn: HookFn) {
    (this.hooks[kind] as HookFn[]).push(fn);
  }

  public getHooks(): TestHooks {
    return this.hooks;
  }
}

export default TestRegistry;
