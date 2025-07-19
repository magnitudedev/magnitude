#!/usr/bin/env node
import { Command } from 'commander';
import path from 'node:path';
import fs from 'node:fs';
import { glob } from 'glob';
//import { Magnitude, TestCase } from '..';
//import { LocalTestRunner } from '@/runner';
import { MagnitudeConfig } from '@/discovery/types';
//import chalk from 'chalk';
import { discoverTestFiles, findConfig, findProjectRoot, isProjectRoot, readConfig } from '@/discovery/util';
//import { BaseTestRunner, BaseTestRunnerConfig } from './runner/baseRunner';
import { logger as coreLogger } from 'magnitude-core';
import logger from '@/logger';
import { describeModel } from './util';
import * as dotenv from 'dotenv';
import { execSync } from 'node:child_process';
// Removed React import
// Removed App import
// Removed render import
import { TestSuiteRunner, TestSuiteRunnerConfig } from './runner/testSuiteRunner'; // Import the new executor and config
import { TermAppRenderer } from '@/term-app'; // Import TermAppRenderer
//import { initializeTestStates } from './term-app/util';
// Removed import { initializeUI, updateUI, cleanupUI } from '@/term-app';
import { startWebServers, stopWebServers } from './webServer';
import chalk from 'chalk';
import os from 'os';

interface CliOptions {
    workers?: number;
    plain: boolean;
    debug: boolean;
}

function getRelativePath(projectRoot: string, absolutePath: string): string {
    // Ensure both paths are absolute and normalized
    const normalizedAbsolutePath = path.normalize(absolutePath);
    const normalizedProjectRoot = path.normalize(projectRoot);

    // Check if the path is inside the project root
    if (!normalizedAbsolutePath.startsWith(normalizedProjectRoot)) {
        // If the path is not within the project root, return the original path
        return absolutePath;
    }

    return path.relative(normalizedProjectRoot, normalizedAbsolutePath);
}

const configTemplate = `import { type MagnitudeConfig } from 'magnitude-test';

// Learn more about configuring Magnitude:
// https://docs.magnitude.run/customizing/configuration

export default {
    url: "http://localhost:5173"
} satisfies MagnitudeConfig;
`;

const exampleTestTemplate = `import { test } from 'magnitude-test';

// Learn more about building test case:
// https://docs.magnitude.run/core-concepts/building-test-cases

const sampleTodos = [
    "Take out the trash",
    "Pay AWS bill",
    "Build more test cases with Magnitude"
];

test('can add and complete todos', { url: 'https://magnitodo.com' }, async (agent) => {
    await agent.act('create 3 todos', { data: sampleTodos.join(', ') });
    await agent.check('should see all 3 todos');
    await agent.act('mark each todo complete');
    await agent.check('says 0 items left');
});
`;

async function initializeProject(force = false, destination = 'tests/magnitude'): Promise<void> {
    /**
     * Initialize magnitude test case files in a node project
     */
    const cwd = process.cwd();
    const isNodeProject = await isProjectRoot(cwd);

    if (!isNodeProject && !force) {
        console.error("Couldn't find package.json in current directory, please initialize Magnitude in a node.js project");
        console.error("To override this check, use --force option");
        process.exit(1);
    }

    console.log(chalk.blueBright(`Initializing Magnitude tests in ${cwd}`));

    // Create directory structure
    const testsDir = path.join(cwd, destination);

    const configPath = path.join(testsDir, 'magnitude.config.ts');

    if (fs.existsSync(configPath)) {
        console.error("Already initialized, magnitude.config.ts already exists!");
        process.exit(1);
    }

    try {
        // Create directories recursively
        await fs.promises.mkdir(testsDir, { recursive: true });

        // Create config file
        await fs.promises.writeFile(configPath, configTemplate);

        // Create example test file
        const examplePath = path.join(testsDir, 'example.mag.ts');
        await fs.promises.writeFile(examplePath, exampleTestTemplate);

        console.log(`${chalk.blueBright('✓')} Created Magnitude test directory structure:
    - ${path.relative(cwd, configPath)}
    - ${path.relative(cwd, examplePath)}
  `);

    } catch (error) {
        console.error('Error initializing Magnitude project:', error);
        process.exit(1);
    }

    // Run Playwright installation for Chromium
    console.log(chalk.blueBright('Installing Playwright Chromium...'));
    try {
        execSync('npx playwright install chromium', { stdio: 'inherit' });
        console.log(`${chalk.blueBright('✓')} Playwright Chromium installed successfully`);
    } catch (error) {
        console.error('Error installing Playwright Chromium:', error);
        // Don't exit with error code since the initialization succeeded
        console.log(chalk.blueBright('You may need to manually run: npx playwright install chromium'));
    }

    console.log(`You can now run tests with: ${chalk.blueBright('npx magnitude')}`);
    console.log('Docs:', chalk.blueBright('https://docs.magnitude.run'));
}

const program = new Command();

program
    .name('magnitude')
    .description('Run Magnitude test cases')
    .argument('[filter]', 'glob pattern for test files (quote if contains spaces or wildcards)')
    .option('-w, --workers <number>', 'number of parallel workers for test execution', '1')
    .option('-p, --plain', 'disable pretty output and print lines instead')
    .option('-d, --debug', 'enable debug logs')
    // Changed action signature from (filters, options) to (filter, options)
    .action(async (filter, options: CliOptions) => {
        dotenv.config();
        let logLevel: string;

        if (process.env.MAGNITUDE_LOG_LEVEL) {
            logLevel = process.env.MAGNITUDE_LOG_LEVEL;
        } else if (options.debug) {
            logLevel = 'trace';
        } else if (options.plain) {
            // TODO: have distinct / nicer clean logs for plain output instead of just changing log level
            logLevel = 'info';
        } else {
            logLevel = 'warn';
        }
        coreLogger.level = logLevel;
        logger.level =logLevel;

        const patterns = [
            '!**/node_modules/**',
            '!**/dist/**'
        ];

        if (filter) {
            patterns.push(filter);
        } else {
            // Default pattern if no filter is provided
            patterns.push('**/*.{mag,magnitude}.{js,jsx,ts,tsx}');
        }

        const workerCount = options.workers ? parseInt(options.workers as unknown as string, 10) : 1;
        if (isNaN(workerCount) || workerCount < 1) {
            console.error('Invalid worker count. Using default of 1.');
        }

        const absoluteFilePaths = await discoverTestFiles(patterns);

        if (absoluteFilePaths.length === 0) {
            console.error(`No test files found matching patterns: ${patterns.join(', ')}`);
            process.exit(1);
        }
        // only matters to show file names nicely
        const projectRoot = await findProjectRoot() ?? process.cwd();

        const configPath = findConfig(projectRoot);

        //console.log(configPath)

        const config: MagnitudeConfig = configPath ? await readConfig(configPath) : {};

        //console.log(config)

        // Check if any API key is set in environment variables or config file
        const hasApiKey = process.env.ANTHROPIC_API_KEY ||
        (config.llm && 'options' in config.llm && 'apiKey' in config.llm.options && config.llm.options.apiKey);
        
        // Check if user is authenticated through Claude Code
        let hasClaudeCodeAuth = false;
        try {
            const CREDS_PATH = path.join(os.homedir(), '.magnitude', 'credentials', 'claudeCode.json');
            const data = await fs.promises.readFile(CREDS_PATH, 'utf-8');
            const creds = JSON.parse(data);
            hasClaudeCodeAuth = creds.expires_at > Date.now() + 60000;
        } catch (error) {
            hasClaudeCodeAuth = false;
        }
        
        if (!hasApiKey && !hasClaudeCodeAuth) {
            console.error("Missing API key for LLM provider");
            console.error("You can either:");
            console.error("  - Set ANTHROPIC_API_KEY environment variable");
            console.error("  - Configure an API key in your magnitude.config.ts");
            console.error("  - Authenticate with Claude Code");
            process.exit(1);
        }

        // // If planner not provided, make a choice based on available environment variables
        // if (!config.planner) {
        //     const planner = tryDeriveEnvironmentPlannerClient();
        //     if (!planner) {
        //         // TODO: Should point to docs on configuration
        //         console.error("No planner client configured. Set an appropriate environment variable or configure planner in magnitude.config.ts");
        //         process.exit(1);
        //     }
        //     config.planner = planner;
        // }

        // logger.info({ ...config.planner }, "Planner:");
        // //console.log(magnitudeBlue(`Using planner: ${describeModel(config.planner)}`));

        // // If executor not provided, default to moondream cloud with MOONDREAM_API_KEY
        // if (!config.executor || !config.executor.options || (!config.executor.options.apiKey && !config.executor.options.baseUrl)) {
        //     const apiKey = process.env.MOONDREAM_API_KEY;
        //     if (!apiKey) {
        //         console.error("Missing MOONDREAM_API_KEY, get one at https://moondream.ai/c/cloud/api-keys");
        //         process.exit(1);
        //     }

        //     config.executor = {
        //         provider: 'moondream',
        //         options: {
        //             apiKey
        //             // don't pass base URL, use moondream client default (https://api.moondream.ai/v1)
        //         }
        //     }
        // }

        // logger.info({ ...config.executor }, "Executor:");
        //console.log(magnitudeBlue(`Using executor: ${config.executor.provider}`));

        let webServerProcesses: (import('node:child_process').ChildProcess | null)[] = [];
        if (config.webServer) {
            try {
                webServerProcesses = await startWebServers(config.webServer);
                const cleanup = () => stopWebServers(webServerProcesses);
                process.on('exit', cleanup);
                process.on('SIGINT', () => { cleanup(); process.exit(1); });
            } catch (err) {
                console.error('Error starting web server(s):', err);
                process.exit(1);
            }
        }


        const showUI = !options.debug && !options.plain;


        const testSuiteRunner = new TestSuiteRunner({
            config,
            workerCount: workerCount,
            createRenderer: (tests) => showUI
                ? new TermAppRenderer(config, tests)
                : {
                    // Plain/debug renderer
                    onTestStateUpdated: (test, state) => {
                        logger.info(`Test: ${test.title} (${test.id})`);
                        logger.info(`  Status: ${state.failure ? 'failed' : (state.doneAt ? 'passed' : (state.startedAt ? 'running' : 'pending'))}`);
                        if (state.failure) {
                            logger.error(`  Failure: ${state.failure.message}`);
                        }
                    }
                },
        });

        for (const filePath of absoluteFilePaths) {
            await testSuiteRunner.loadTestFile(filePath, getRelativePath(projectRoot, filePath));
        }

        try {
            const overallSuccess = await testSuiteRunner.runTests();
            process.exit(overallSuccess ? 0 : 1);
        } catch (error) {
            logger.error("Test suite execution failed:", error);
            process.exit(1);
        }
    });

program
    .command('init')
    .description('Initialize Magnitude test directory structure')
    .option('-f, --force', 'force initialization even if no package.json is found')
    .option('--dir, --destination <path>', 'destination directory for Magnitude tests', 'tests/magnitude')
    .action(async (options) => {
        await initializeProject(options.force, options.destination);
    });

program.parse();
