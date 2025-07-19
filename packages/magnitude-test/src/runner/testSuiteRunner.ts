import { MagnitudeConfig, RegisteredTest, TestOptions } from '@/discovery/types';
import { processUrl } from '../util';
import { WorkerPool, WorkerPoolResult } from './workerPool';
import { TestResult, TestState } from './state';
import { TestRenderer } from '@/renderer';
import { Worker } from 'node:worker_threads';
import { TestWorkerData, TestWorkerIncomingMessage, TestWorkerOutgoingMessage } from '@/worker/util';
import { isBun, isDeno } from 'std-env';
import { EventEmitter } from 'node:events';

const TEST_FILE_LOADING_TIMEOUT = 30000;

export interface TestSuiteRunnerConfig {
    workerCount: number;
    createRenderer: (tests: RegisteredTest[]) => TestRenderer;
    config: MagnitudeConfig;
}

export class TestSuiteRunner {
    private runnerConfig: TestSuiteRunnerConfig;
    private renderer?: TestRenderer;
    private config: MagnitudeConfig;

    private tests: RegisteredTest[];
    private executors: Map<string, ClosedTestExecutor> = new Map();
    private workerAborters: AbortController[] = [];

    constructor(
        config: TestSuiteRunnerConfig
    ) {
        this.tests = [];
        this.runnerConfig = config;
        this.config = config.config;
    }

    private runTest(test: RegisteredTest, signal: AbortSignal): Promise<TestResult> {
        const executor = this.executors.get(test.id);
        if (!executor) {
            throw new Error(`Test worker not found for test ID: ${test.id}`);
        }

        return executor(
            {
                type: "execute",
                test,
                browserOptions: this.config.browser,
                llm: this.config.llm,
                grounding: this.config.grounding,
                telemetry: this.config.telemetry
            },
            (state: TestState) => {
                this.renderer?.onTestStateUpdated(test, state);
            },
            signal
        );
    }

    private getActiveOptions(): TestOptions {
        const envOptions = process.env.MAGNITUDE_TEST_URL ? {
            url: process.env.MAGNITUDE_TEST_URL
        } : {};

        return {
            ...this.config,
            ...envOptions, // env options take precedence over config options
            url: processUrl(envOptions.url, this.config.url),
        };
    }

    public async loadTestFile(absoluteFilePath: string, relativeFilePath: string): Promise<void> {
        try {
            const workerData = {
                relativeFilePath,
                absoluteFilePath,
                options: this.getActiveOptions(),
            } satisfies TestWorkerData;

            const createWorker = isBun ? createBunTestWorker : createNodeTestWorker;
            const result = await createWorker(workerData);

            this.tests.push(...result.tests);
            for (const test of result.tests) {
                this.executors.set(test.id, result.executor);
            }
            this.workerAborters.push(result.aborter);
        } catch (error) {
            console.error(`Failed to load test file ${relativeFilePath}:`, error);
            throw error;
        }
    }

    async runTests(): Promise<boolean> {
        if (!this.tests) throw new Error('No tests were registered');
        this.renderer = this.runnerConfig.createRenderer(this.tests);
        this.renderer.start?.();
        const workerPool = new WorkerPool(this.runnerConfig.workerCount);

        const taskFunctions = this.tests.map((test) => {
            return async (signal: AbortSignal): Promise<TestResult> => {
                try {
                    return await this.runTest(test, signal);
                } catch (err: unknown) {
                    // user-facing, can happen e.g. when URL is not running
                    if (err instanceof Error) {
                        console.error(`Unexpected error during test '${test.title}':\n${err.message}`);
                    } else {
                        console.error(`Unexpected error during test '${test.title}':\n${err}`);
                    }

                    throw err;
                }
            };
        });

        let overallSuccess = true;
        try {
            const poolResult: WorkerPoolResult<TestResult> = await workerPool.runTasks<TestResult>(
                taskFunctions,
                (taskOutcome: TestResult) => !taskOutcome.passed
            );

            for (const result of poolResult.results) {
                if (result === undefined || !result.passed) {
                    overallSuccess = false;
                    break;
                }
            }
            if (!poolResult.completed) { // If pool aborted for any reason (incl. a task failure)
                overallSuccess = false;
            }

        } catch (error) {
            overallSuccess = false;
        }
        for (const aborter of this.workerAborters) {
            aborter.abort();
        }
        this.renderer.stop?.();
        return overallSuccess;

    }
}

type ClosedTestExecutor =
    (
        message: TestWorkerIncomingMessage,
        onStateChange: (state: TestState) => void,
        signal: AbortSignal
    ) => Promise<TestResult>;

type CreateTestWorker = (workerData: TestWorkerData) =>
    Promise<{
        tests: RegisteredTest[];
        executor: ClosedTestExecutor;
        aborter: AbortController;
    }>;

const createNodeTestWorker: CreateTestWorker = async (workerData) =>
    new Promise((resolve, reject) => {
        const { relativeFilePath } = workerData;
        const worker = new Worker(
            new URL(
                import.meta.url.endsWith(".ts")
                    ? '../worker/readTest.ts'
                    : './worker/readTest.js',
                import.meta.url
            ),
            {
                workerData,
                env: { NODE_ENV: 'test', ...process.env },
                execArgv: !(isBun || isDeno) ? ["--import=jiti/register"] : []
            }
        );

        const aborter = new AbortController();
        aborter.signal.addEventListener('abort', () => {
            worker.terminate();
        });

        const executor: ClosedTestExecutor =
            (executeMessage, onStateChange, signal) => new Promise((res, rej) => {
                const messageHandler = (msg: TestWorkerOutgoingMessage) => {
                    if (!("testId" in msg) || msg.testId !== executeMessage.test.id) return;

                    if (msg.type === "test_result") {
                        worker.off("message", messageHandler);
                        res(msg.result);
                    } else if (msg.type === "test_error") {
                        worker.off("message", messageHandler);
                        rej(new Error(msg.error));
                    } else if (msg.type === "test_state_change") {
                        onStateChange(msg.state);
                    }
                };

                signal.addEventListener('abort', () => {
                    worker.off("message", messageHandler);
                    rej(new Error('Test execution aborted'));
                });

                worker.on("message", messageHandler);
                worker.postMessage(executeMessage);
            });

        const registeredTests: RegisteredTest[] = [];

        worker.on('message', (message: TestWorkerOutgoingMessage) => {
            if (message.type === 'registered') {
                registeredTests.push(message.test);

            } else if (message.type === 'load_error') {
                clearTimeout(timeout);
                worker.terminate();
                reject(new Error(`Failed to load ${relativeFilePath}: ${message.error}`));
            } else if (message.type === 'load_complete') {
                clearTimeout(timeout);
                if (!registeredTests.length) {
                    reject(new Error(`No tests registered for file ${relativeFilePath}`));
                    return;
                }
                resolve({ tests: registeredTests, executor, aborter });
            }
        });

        const timeout = setTimeout(() => {
            worker.terminate();
            reject(new Error(`Test file loading timeout: ${relativeFilePath}`));
        }, TEST_FILE_LOADING_TIMEOUT);

        worker.on('error', (error) => {
            clearTimeout(timeout);
            worker.terminate();
            reject(error);
        });
    });

const createBunTestWorker: CreateTestWorker = async (workerData) =>
    new Promise((resolve, reject) => {
        const { relativeFilePath } = workerData;
        const emit = new EventEmitter();
        const proc = Bun.spawn({
            cmd: [
                "bun",
                new URL(
                    import.meta.url.endsWith(".ts")
                        ? '../worker/readTest.ts'
                        : './worker/readTest.js',
                    import.meta.url
                ).pathname
            ],
            env: {
                NODE_ENV: 'test',
                ...process.env,
                MAGNITUDE_WORKER_DATA: JSON.stringify(workerData)
            },
            cwd: process.cwd(),
            stdin: "inherit",
            stdout: "inherit",
            stderr: "inherit",
            // "advanced" serialization in Bun somehow isn't able to clone test state messages?
            serialization: 'json',
            ipc(message) {
                emit.emit('message', message);
            },
            onExit(_subprocess, exitCode) {
                if (exitCode !== 0) {
                    clearTimeout(timeout);
                    reject(new Error(`Worker process exited with code ${exitCode}`));
                }
            },
        });

        const aborter = new AbortController();
        aborter.signal.addEventListener('abort', () => {
            proc.kill("SIGKILL");
        });

        const executor: ClosedTestExecutor = (executeMessage, onStateChange, signal) =>
            new Promise((res, rej) => {
                const messageHandler = (msg: TestWorkerOutgoingMessage) => {
                    if (!("testId" in msg) || msg.testId !== executeMessage.test.id) return;

                    if (msg.type === "test_result") {
                        emit.off('message', messageHandler);
                        res(msg.result);
                    } else if (msg.type === "test_error") {
                        emit.off('message', messageHandler);
                        rej(new Error(msg.error));
                    } else if (msg.type === "test_state_change") {
                        onStateChange(msg.state);
                    }
                };

                emit.on('message', messageHandler);

                signal.addEventListener('abort', () => {
                    emit.off('message', messageHandler);
                    rej(new Error('Test execution aborted'));
                });

                proc.send(executeMessage);

            });

        const registeredTests: RegisteredTest[] = [];

        emit.on('message', ((message: TestWorkerOutgoingMessage) => {
            if (message.type === 'registered') {
                registeredTests.push(message.test);
            } else if (message.type === 'load_error') {
                clearTimeout(timeout);
                proc.kill();
                reject(new Error(`Failed to load ${relativeFilePath}: ${message.error}`));
            } else if (message.type === 'load_complete') {
                clearTimeout(timeout);
                if (!registeredTests.length) {
                    reject(new Error(`No tests registered for file ${relativeFilePath}`));
                    return;
                }
                resolve({ tests: registeredTests, executor, aborter });
            }
        }));

        const timeout = setTimeout(() => {
            proc.kill("SIGKILL");
            reject(new Error(`Test file loading timeout: ${relativeFilePath}`));
        }, TEST_FILE_LOADING_TIMEOUT);
    });
