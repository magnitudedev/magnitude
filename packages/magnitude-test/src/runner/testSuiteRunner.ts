import { MagnitudeConfig, RegisteredTest, TestOptions } from '@/discovery/types';
import { processUrl } from '../util';
import { WorkerPool } from './workerPool';
import { TestResult, TestState } from './state';
import { TestRenderer } from '@/renderer';
import { Worker } from 'node:worker_threads';
import { TestWorkerData, TestWorkerIncomingMessage, TestWorkerOutgoingMessage } from '@/worker/util';
import { isBun, isDeno } from 'std-env';
import { EventEmitter } from 'node:events';

const TEST_FILE_LOADING_TIMEOUT = 30000;
const WORKER_SHUTDOWN_TIMEOUT = 60000;

export interface TestSuiteRunnerConfig {
    workerCount: number;
    createRenderer: (tests: RegisteredTest[]) => TestRenderer;
    config: MagnitudeConfig;
    failFast?: boolean;
}

export class TestSuiteRunner {
    private runnerConfig: TestSuiteRunnerConfig;
    private renderer?: TestRenderer;
    private config: MagnitudeConfig;

    private tests: RegisteredTest[];
    private executors: Map<string, ClosedTestExecutor> = new Map();
    private workerStoppers: (() => Promise<void>)[] = [];

    constructor(
        config: TestSuiteRunnerConfig
    ) {
        this.tests = [];
        this.runnerConfig = config;
        this.config = config.config;
    }

    private async runTest(test: RegisteredTest, signal: AbortSignal): Promise<TestResult> {
        const executor = this.executors.get(test.id);
        if (!executor) {
            throw new Error(`Test worker not found for test ID: ${test.id}`);
        }

        try {
            return await executor(
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
        } catch (err: unknown) {
            if (signal.aborted) {
                return {
                    passed: false,
                    failure: {
                        message: 'Test execution aborted'
                    }
                };
            }

            const errorMessage = err instanceof Error ? err.message : String(err);
            // user-facing, can happen e.g. when URL is not running
            console.error(`Unexpected error during test '${test.title}':\n${errorMessage}`);
            return {
                passed: false,
                failure: {
                    message: errorMessage
                }
            };
        }
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
            this.workerStoppers.push(result.stopper);
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

        let overallSuccess = true;
        try {
            const poolResult = await workerPool.runTasks(
                this.tests.map((test) => (signal) => this.runTest(test, signal)),
                this.runnerConfig.failFast
                    ? (taskOutcome: TestResult) => !taskOutcome.passed
                    : () => false
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

        const stopperResults = await Promise.allSettled(
            this.workerStoppers.map(stopper => stopper())
        );

        const stopperErrors = stopperResults
            .filter(result => result.status === 'rejected');

        if (stopperErrors.length > 0) {
            overallSuccess = false;
            console.error(`${stopperErrors.length} workers failed to stop`);
            for (const error of stopperErrors) {
                console.error(error);
            }
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
        stopper: () => Promise<void>;
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

        let hasRunTests = false;
        const stopper = async (): Promise<void> => {
            if (!hasRunTests) {
                worker.terminate();
                return;
            }

            worker.postMessage({ type: 'graceful_shutdown' });

            return new Promise((resolve, reject) => {
                const timeout = setTimeout(() => {
                    worker.off('message', shutdownHandler);
                    worker.terminate();
                    reject(new Error('graceful shutdown timeout'));
                }, WORKER_SHUTDOWN_TIMEOUT);

                const shutdownHandler = (msg: TestWorkerOutgoingMessage) => {
                    if (msg.type === 'graceful_shutdown_complete') {
                        clearTimeout(timeout);
                        worker.off('message', shutdownHandler);
                        worker.terminate();
                        resolve();
                    }
                };

                worker.on('message', shutdownHandler);
            });
        };

        const executor: ClosedTestExecutor =
            (executeMessage, onStateChange, signal) => new Promise((res, rej) => {
                const messageHandler = (msg: TestWorkerOutgoingMessage) => {
                    if ("test" in executeMessage && "testId" in msg && msg.testId !== executeMessage.test.id) return;
                    hasRunTests = true;

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
                resolve({ tests: registeredTests, executor, stopper });
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

        let hasRunTests = false;
        const stopper = async (): Promise<void> => {
            if (!hasRunTests) {
                proc.kill("SIGKILL");
                return;
            }

            proc.send({ type: 'graceful_shutdown' });

            return new Promise((resolve, reject) => {
                const timeout = setTimeout(() => {
                    emit.off('message', shutdownHandler);
                    proc.kill("SIGKILL");
                    reject(new Error('graceful shutdown timeout'));
                }, WORKER_SHUTDOWN_TIMEOUT);

                const shutdownHandler = (msg: TestWorkerOutgoingMessage) => {
                    if (msg.type === 'graceful_shutdown_complete') {
                        clearTimeout(timeout);
                        emit.off('message', shutdownHandler);
                        proc.kill("SIGKILL");
                        resolve();
                    }
                };

                emit.on('message', shutdownHandler);
            });
        };

        const executor: ClosedTestExecutor = (executeMessage, onStateChange, signal) =>
            new Promise((res, rej) => {
                const messageHandler = (msg: TestWorkerOutgoingMessage) => {
                    if ("test" in executeMessage && "testId" in msg && msg.testId !== executeMessage.test.id) return;
                    hasRunTests = true;

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
                resolve({ tests: registeredTests, executor, stopper });
            }
        }));

        const timeout = setTimeout(() => {
            proc.kill("SIGKILL");
            reject(new Error(`Test file loading timeout: ${relativeFilePath}`));
        }, TEST_FILE_LOADING_TIMEOUT);
    });
