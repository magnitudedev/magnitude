import { BrowserContext, BrowserContextOptions, LaunchOptions, Page } from "playwright";
import type { PlannerClient, ExecutorClient, TestCaseAgent, Magnus } from 'magnitude-core';

export interface TestOptions {
    url?: string;
    //name?: string;
}

export interface WebServerConfig {
    command: string;
    url: string;
    timeout?: number;
    reuseExistingServer?: boolean;
}

export type MagnitudeConfig = {
    //apiKey?: string;
    url: string; // base URL used as default, required
    planner?: PlannerClient,
    executor?: ExecutorClient,
    webServer?: WebServerConfig,
    browser?: {
        contextOptions?: BrowserContextOptions,
        launchOptions?: LaunchOptions
    },
    telemetry?: boolean,
    display?: {
        showActions?: boolean;
    },
    downscaling?: number;
}

export interface TestFunctionContext {
    ai: Magnus;
    get page(): Page;
    get context(): BrowserContext;
    //page: Page;
    //context: Context;
}

export type TestFunction = (context: TestFunctionContext) => Promise<void>;
export type TestGroupFunction = () => void;

export interface TestRunnable {
    fn: TestFunction
    title: string
    url: string
}

export type CategorizedTestRunnable = TestRunnable & { file: string, group: string | null };

export interface TestGroup {
    name: string;
    options?: TestOptions;
}

export interface TestGroupDeclaration {
    (id: string, options: TestOptions, groupFn: TestGroupFunction): void;
    (id: string, groupFn: TestGroupFunction): void;
}

export interface TestDeclaration {
    (title: string, options: TestOptions, testFn: TestFunction): void;
    (title: string, testFn: TestFunction): void;
    //(id: string, options?: TestOptions): TestCaseBuilder;
    //(title: string, options?: TestOptions): void;

    group: TestGroupDeclaration;
}

// Map from filepath to grouped and ungrouped test cases
export type CategorizedTestCases = Record<string, { ungrouped: TestRunnable[], groups: Record<string, TestRunnable[]>}>;
//export type CategorizedTestCases = Record<string, { ungrouped: TestCaseBuilder[], groups: Record<string, TestCaseBuilder[]>}>;
