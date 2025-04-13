import { executeCliCommand, handleError } from './utils/cliUtils.js';
import { logger } from './utils/logger.js';
import { InitializeProjectInput, RunTestsInput } from './types.js';

/**
 * Initialize a new Magnitude project
 * @param args Arguments for initializing project
 * @returns MCP response
 */
export async function initializeProject(args: InitializeProjectInput): Promise<any> {
    const { projectDir } = args;
    logger.info('[Setup] Initializing Magnitude project...');

    try {
        // Use the Magnitude CLI with spawn approach
        const installOutput = await executeCliCommand('npm', ['install', 'magnitude-test'], { cwd: projectDir });
        const initOutput = await executeCliCommand('npx', ['magnitude', 'init'], { cwd: projectDir });

        logger.info('[Setup] Magnitude project initialized successfully');

        return {
            content: [
                {
                    type: 'text',
                    text: `${installOutput}\n\n${initOutput}\nMagnitude project initialized successfully.`,
                },
            ],
        };
    } catch (error) {
        return handleError('Failed to initialize project', error);
    }
}

/**
 * Run Magnitude tests
 * @param args Arguments for running tests
 * @returns MCP response
 */
export async function runTests(args: RunTestsInput): Promise<any> {
    logger.info('[Test] Running Magnitude tests');

    try {
        const { projectDir, pattern, workers } = args;

        // Build command arguments
        const cmdArgs = ['magnitude'];

        if (pattern) {
            cmdArgs.push(pattern);
        }

        if (workers && Number.isInteger(workers) && workers > 0) {
            cmdArgs.push('-w', workers.toString());
        }

        logger.info(`[Test] Executing command: npx ${cmdArgs.join(' ')} in ${projectDir}`);

        // Execute command
        try {
            const output = await executeCliCommand('npx', cmdArgs, {
                cwd: projectDir // This handles the directory change
            });

            return {
                content: [
                    {
                        type: 'text',
                        text: `Tests executed successfully:\n\n${output}`,
                    },
                ],
            };
        } catch (error: any) {
            // If the tests fail, the process will exit with a non-zero code
            // But we still want to return the output
            return {
                content: [
                    {
                        type: 'text',
                        text: `Tests executed with failures:\n\n${error.message || ''}`,
                    },
                ],
                isError: true,
            };
        }
    } catch (error) {
        return handleError('Failed to run tests', error);
    }
}

/**
 * Build test cases by fetching documentation on how to design proper Magnitude test cases
 * @returns MCP response with formatted documentation
 */
export async function buildTests(): Promise<any> {
    logger.info('[Build] Fetching Magnitude test case documentation');

    try {
        // Fetch the LLMs full text file
        const response = await fetch("https://docs.magnitude.run/llms-full.txt");

        if (!response.ok) {
            throw new Error(`Failed to fetch documentation: ${response.status} ${response.statusText}`);
        }

        const fullText = await response.text();

        // Find the start of the "## Test Cases" section instead of "# Building Test Cases"
        const buildingTestCasesIndex = fullText.indexOf("# Building Test Cases");
        const testCasesIndex = fullText.indexOf("## Test Cases", buildingTestCasesIndex);

        // Use testCasesIndex as the starting point
        const startIndex = testCasesIndex;

        // Find the start of the "Example of migrating a Playwright test case to Magnitude" section
        // which is where we want to end our extraction
        const exampleSectionIndex = fullText.indexOf("### Example of migrating a Playwright test case to Magnitude", startIndex);

        // Extract the content from "## Test Cases" to the start of the example section
        const content = fullText.substring(startIndex, exampleSectionIndex).trim();

        // Add the introductory text at the beginning and the concluding text at the end with markdown formatting
        const introText = "This is the section from the Magnitude docs on how to design proper test cases:\n\n";

        // Add an important note about login requirements
        const loginNote = "\n\n## IMPORTANT NOTE:\n\n" +
            "If the user's site requires login, then **EVERY test case** will need to start with a login step with proper data attached.\n\n";

        const concludingText = "## Now that you know how to build proper Magnitude test cases, build test cases for the user for whatever they are asking about.\n\n" +
            "- Put the test cases in a **new** .mag.ts file if building a fresh page/feature, or edit the relevant **existing** .mag.ts file if expanding on an existing page/feature.\n\n" +
            "- Follow the Magnitude docs **extremely closely** when building test cases.\n" +
            "- Do not overcomplicate. Keep the test cases **simple and straightforward**.\n" +
            "- Do not write too many test cases. Just cover the **main flows** for whatever the user is asking about.";

        const formattedContent = introText + content + loginNote + concludingText;

        return {
            content: [
                {
                    type: 'text',
                    text: formattedContent,
                },
            ],
        };
    } catch (error) {
        return handleError('Failed to fetch test case documentation', error);
    }
} 