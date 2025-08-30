import { setImmediate } from "node:timers";
import logUpdate from 'log-update';
import { RegisteredTest } from '@/discovery/types';
import { TestFailure, TestState as RunnerTestState, StepDescriptor as RunnerStepDescriptor, CheckDescriptor as RunnerCheckDescriptor } from '@/runner/state';
import { AllTestStates } from './types';
import { VERSION } from '@/version';
import { formatDuration } from './util'; // wrapText removed
import {
    ANSI_RESET, ANSI_GREEN, ANSI_BRIGHT_BLUE,
    ANSI_GRAY, ANSI_RED, ANSI_BOLD, ANSI_DIM
} from './constants'; // PADDING, BOX_CHARS_ROUNDED removed
import {
    str, styleAnsi,
    getTestStatusIndicatorChar,
    getStepStatusIndicatorChar, getCheckStatusIndicatorChar
} from './drawingUtils'; // createBoxAnsi, insertLineIntoBoxAnsi removed
import {
    // currentWidth, // Will be unused if all wrapping is gone (now removed from uiState)
    redrawScheduled, currentTestStates, allRegisteredTests,
    currentModel, elapsedTimes, isFinished, spinnerFrame,
    lastOutputLineCount, isFirstDraw, /* isResizing, */ renderSettings, // isResizing removed from uiState
    setRedrawScheduled, setLastOutputLineCount, setIsFirstDraw, spinnerChars
} from './uiState';
import { knownCostMap } from '@/util';

const UI_LEFT_PADDING = '  ';

/**
 * Generate the title bar portion of the UI
 * @returns Array of strings with ANSI codes representing the title bar
 */
export function generateTitleBarString(): string[] {
    const titleText = `${ANSI_BRIGHT_BLUE}${ANSI_BOLD}Magnitude v${VERSION}${ANSI_RESET}`;
    const modelText = `${ANSI_GRAY}${currentModel}${ANSI_RESET}`;
    // Simple single line for title bar, no complex padding or width calculations
    return [`${UI_LEFT_PADDING}${titleText}  ${modelText}`];
}

/**
 * Generate a string representation of a failure
 */
export function generateFailureString(failure: TestFailure, indent: number): string[] {
    const output: string[] = [];
    const prefix = '↳ ';
    const prefixAnsi = `${ANSI_RED}${prefix}${ANSI_RESET}`;

    const addLine = (text: string, styleCode = ANSI_RED, bold = false) => {
        const fullStyleCode = `${styleCode}${bold ? ANSI_BOLD : ''}`;
        // No wrapping, text is a single line
        output.push(UI_LEFT_PADDING + ' '.repeat(indent) + prefixAnsi + `${fullStyleCode}${text}${ANSI_RESET}`);
    };
    
    if (failure && failure.message) {
        addLine(failure.message);
    } else {
        addLine("Unknown error details");
    }
    return output;
}

/**
 * Generate a string representation of a test
 */
export function generateTestString(test: RegisteredTest, state: RunnerTestState, indent: number): string[] {
    const output: string[] = [];
    const testId = test.id;
    const stepIndent = indent + 2;
    const actionIndent = stepIndent + 2;
    
    const currentStatus = state.status;
    const statusCharPlain = currentStatus === 'running' ? spinnerChars[spinnerFrame] : getTestStatusIndicatorChar(currentStatus);
    const statusStyled = styleAnsi(currentStatus, statusCharPlain, 'test');
    const timerText = currentStatus !== 'pending' ? `${ANSI_GRAY} [${formatDuration(elapsedTimes[testId] ?? 0)}]${ANSI_RESET}` : '';
    
    // No wrapping for title
    output.push(UI_LEFT_PADDING + ' '.repeat(indent) + `${statusStyled} ${test.title}${timerText}`);

    if (state.stepsAndChecks && state.stepsAndChecks.length > 0) {
        state.stepsAndChecks.forEach((item: RunnerStepDescriptor | RunnerCheckDescriptor) => {
            let itemCharPlain = '';
            let itemDesc = '';
            let itemStyleType: 'step' | 'check' = 'step';

            if (item.variant === 'step') {
                itemCharPlain = getStepStatusIndicatorChar(item.status);
                itemDesc = item.description;
                itemStyleType = 'step';
            } else { // Check
                itemCharPlain = getCheckStatusIndicatorChar(item.status);
                itemDesc = item.description;
                itemStyleType = 'check';
            }

            const styledChar = styleAnsi(item.status, itemCharPlain, itemStyleType);
            // No wrapping for description
            output.push(UI_LEFT_PADDING + ' '.repeat(stepIndent) + `${styledChar} ${itemDesc}`);

            if (renderSettings.showActions && item.variant === 'step' && (item as RunnerStepDescriptor).actions.length > 0) {
                (item as RunnerStepDescriptor).actions.forEach((action) => {
                    // const actionSymbol = `${ANSI_GRAY}▶${ANSI_RESET}`; 
                    // const actionDesc = JSON.stringify(action);
                    // No wrapping for action description
                    output.push(UI_LEFT_PADDING + ' '.repeat(actionIndent) + `${ANSI_GRAY}${action.pretty}${ANSI_RESET}`);//`${actionSymbol} ${ANSI_GRAY}${actionDesc}${ANSI_RESET}`);
                });
            }
        });
    }

    if (state.failure) {
        const failureLines = generateFailureString(state.failure, stepIndent);
        output.push(...failureLines);
    }
    return output;
}

// Tree node structure for hierarchical display
interface TreeNode {
    tests: RegisteredTest[];
    children: Record<string, TreeNode>;
}

// Helper function to build a tree structure from hierarchical tests
function buildTestTree(tests: RegisteredTest[]): TreeNode {
    const root: TreeNode = { tests: [], children: {} };

    for (const test of tests) {
        if (!test.groupHierarchy || test.groupHierarchy.length === 0) {
            // Tests without hierarchy go in the root
            root.tests.push(test);
        } else {
            // Build the tree path for hierarchical tests
            let currentNode = root;
            for (const group of test.groupHierarchy) {
                if (!currentNode.children[group.name]) {
                    currentNode.children[group.name] = { tests: [], children: {} };
                }
                currentNode = currentNode.children[group.name];
            }
            currentNode.tests.push(test);
        }
    }

    return root;
}

// Helper function to group tests by file and build trees
function groupTestsByFile(tests: RegisteredTest[]): Record<string, TreeNode> {
    const files: Record<string, RegisteredTest[]> = {};

    // Group tests by file first
    for (const test of tests) {
        if (!files[test.filepath]) {
            files[test.filepath] = [];
        }
        files[test.filepath].push(test);
    }

    // Build tree for each file
    const fileTrees: Record<string, TreeNode> = {};
    for (const [filepath, fileTests] of Object.entries(files)) {
        fileTrees[filepath] = buildTestTree(fileTests);
    }

    return fileTrees;
}


// Helper function to recursively generate tree display
function generateTreeDisplay(node: TreeNode, indent: number, output: string[]): void {
    // Display tests at this level
    for (const test of node.tests) {
        const state = currentTestStates[test.id];
        if (state) {
            const testLines = generateTestString(test, state, indent);
            output.push(...testLines);
        }
    }

    // Display child groups
    for (const [groupName, childNode] of Object.entries(node.children)) {
        const groupHeader = `${ANSI_BRIGHT_BLUE}${ANSI_BOLD}↳ ${groupName}${ANSI_RESET}`;
        output.push(UI_LEFT_PADDING + ' '.repeat(indent) + groupHeader);

        // Recursively display child node with increased indent
        generateTreeDisplay(childNode, indent + 2, output);
    }
}

/**
 * Generate the test list portion of the UI
 */
export function generateTestListString(): string[] {
    const output: string[] = [];
    const fileIndent = 0;

    const fileTrees = groupTestsByFile(allRegisteredTests);

    for (const [filepath, treeRoot] of Object.entries(fileTrees)) {
        const fileHeader = `${ANSI_BRIGHT_BLUE}${ANSI_BOLD}☰ ${filepath}${ANSI_RESET}`;
        output.push(UI_LEFT_PADDING + ' '.repeat(fileIndent) + fileHeader);

        // Generate tree display starting from root
        generateTreeDisplay(treeRoot, fileIndent + 2, output);

        output.push(UI_LEFT_PADDING); // Blank line between files
    }
    return output;
}

/**
 * Generate the summary portion of the UI
 */
export function generateSummaryString(): string[] {
    const output: string[] = [];
    let totalInputTokens = 0;
    let totalOutputTokens = 0;
    const statusCounts = { pending: 0, running: 0, passed: 0, failed: 0, cancelled: 0, total: 0 };
    const failuresWithContext: { filepath: string; groupName?: string; testTitle: string; failure: TestFailure }[] = [];

    const testContextMap = new Map<string, { filepath: string; groupName?: string; testTitle: string }>();
    allRegisteredTests.forEach(test => {
        testContextMap.set(test.id, { filepath: test.filepath, groupName: test.group, testTitle: test.title });
    });

    Object.entries(currentTestStates).forEach(([testId, state]) => {
        statusCounts.total++;
        statusCounts[state.status]++;
        if (state.modelUsage.length > 0) {
            totalInputTokens += state.modelUsage[0].inputTokens;
            totalOutputTokens += state.modelUsage[0].outputTokens;
        }
        if (state.failure) {
            const context = testContextMap.get(testId);
            failuresWithContext.push({
                filepath: context?.filepath ?? 'Unknown File', groupName: context?.groupName,
                testTitle: context?.testTitle ?? 'Unknown Test', failure: state.failure
            });
        }
    });

    const hasFailures = failuresWithContext.length > 0;

    let statusLine = '';
    if (statusCounts.passed > 0) statusLine += `${ANSI_GREEN}✓ ${statusCounts.passed} passed${ANSI_RESET}  `;
    if (statusCounts.failed > 0) statusLine += `${ANSI_RED}✗ ${statusCounts.failed} failed${ANSI_RESET}  `;
    if (statusCounts.running > 0) statusLine += `${ANSI_BRIGHT_BLUE}▷ ${statusCounts.running} running${ANSI_RESET}  `;
    if (statusCounts.pending > 0) statusLine += `${ANSI_GRAY}◌ ${statusCounts.pending} pending${ANSI_RESET}  `;
    if (statusCounts.cancelled > 0) statusLine += `${ANSI_GRAY}⊘ ${statusCounts.cancelled} cancelled${ANSI_RESET}  `;

    let costDescription = '';
    for (const [model, costs] of Object.entries(knownCostMap)) {
        if (currentModel.includes(model)) {
            const inputCost = costs[0];
            const outputCost = costs[1];
            costDescription = ` (\$${((totalInputTokens * inputCost + totalOutputTokens * outputCost) / 1000000).toFixed(2)})`;
        }
    }
    let tokenText = `${ANSI_GRAY}tokens: ${totalInputTokens} in, ${totalOutputTokens} out${costDescription}${ANSI_RESET}`;

    output.push(UI_LEFT_PADDING + statusLine.trimEnd() + (statusLine && tokenText ? '  ' : '') + tokenText.trimStart());

    if (hasFailures) {
        output.push(UI_LEFT_PADDING + `${ANSI_DIM}Failures:${ANSI_RESET}`);
        for (const { filepath, groupName, testTitle, failure } of failuresWithContext) {
            const contextString = `${ANSI_DIM}${filepath}${groupName ? ` > ${groupName}` : ''} > ${testTitle}${ANSI_RESET}`;
            output.push(UI_LEFT_PADDING + UI_LEFT_PADDING + contextString); // Indent context further with prepended spaces
            const failureLines = generateFailureString(failure, 4); // generateFailureString already adds padding
            output.push(...failureLines);
            output.push(UI_LEFT_PADDING); // Blank line after each failure with prepended spaces
        }
    }
    return output;
}

// Helper function to calculate tree height recursively
function calculateTreeHeight(node: TreeNode, testStates: AllTestStates, indent: number): number {
    let height = 0;

    // Count tests at this level
    for (const test of node.tests) {
        const state = testStates[test.id];
        if (state) {
            height++; // Test title line
            if (state.stepsAndChecks) {
                state.stepsAndChecks.forEach((item: RunnerStepDescriptor | RunnerCheckDescriptor) => {
                    height++; // Item description line
                    if (renderSettings.showActions && item.variant === 'step' && (item as RunnerStepDescriptor).actions.length > 0) {
                        (item as RunnerStepDescriptor).actions.forEach(() => height++);
                    }
                });
            }
            if (state.failure) {
                height++; // generateFailureString returns 1 line
            }
        }
    }

    // Count child groups recursively
    for (const [groupName, childNode] of Object.entries(node.children)) {
        height++; // Group header line
        height += calculateTreeHeight(childNode, testStates, indent + 2);
    }

    return height;
}

/**
 * Calculate the height needed for the test list (now just line count)
 */
export function calculateTestListHeight(tests: RegisteredTest[], testStates: AllTestStates): number {
    let height = 0;
    const fileTrees = groupTestsByFile(tests);

    for (const [filepath, treeRoot] of Object.entries(fileTrees)) {
        height++; // File header line
        height += calculateTreeHeight(treeRoot, testStates, 2); // Start with indent of 2
        height++; // Blank line between files
    }
    return height;
}

/**
 * Calculate the height needed for the summary (now just line count)
 */
export function calculateSummaryHeight(testStates: AllTestStates): number {
    let height = 0;
    height++; // Status counts line

    const failuresExist = Object.values(testStates).some(state => !!state.failure);
    if (failuresExist) {
        height++; // "Failures:" title
        Object.values(testStates).forEach((state) => {
            if (state.failure) {
                height++; // Context line
                height++; // Failure message line (generateFailureString returns 1 line)
                height++; // Blank line after failure
            }
        });
    }
    return height;
}

/**
 * Main function to redraw the UI
 */
export function redraw() {
    setRedrawScheduled(false);

    let testListLineCount = calculateTestListHeight(allRegisteredTests, currentTestStates);
    let summaryLineCount = calculateSummaryHeight(currentTestStates);
    if (Object.values(currentTestStates).length === 0) { // No tests, no summary
        summaryLineCount = 0;
        testListLineCount = 0;
    }

    const outputLines: string[] = [];
    // outputLines.push(''); // Initial blank line for spacing from prompt - REMOVED

    outputLines.push(...generateTitleBarString()); // generateTitleBarString now adds padding
    outputLines.push(UI_LEFT_PADDING); // Blank line after title bar with padding

    if (testListLineCount > 0) {
        outputLines.push(...generateTestListString()); // generateTestListString now adds padding
        // generateTestListString already adds a blank line (now with padding) at the end of each file section
    }

    if (summaryLineCount > 0) {
        if (testListLineCount > 0) outputLines.push(UI_LEFT_PADDING); // Blank line before summary if test list was also shown, with padding
        outputLines.push(...generateSummaryString()); // generateSummaryString now adds padding
    }

    const frameContent = outputLines.join('\n');

    logUpdate.clear(); // Clear previous output before drawing new frame
    logUpdate(frameContent);

    setLastOutputLineCount(outputLines.length); // Still useful for potential future optimizations
    if (isFirstDraw) { // Still useful to track if it's the very first render pass
        setIsFirstDraw(false);
    }

    // If the rendering process has finished (stop() was called),
    // then this redraw is the final one, so call logUpdate.done().
    if (isFinished) {
        logUpdate.done();
        process.stderr.write('\n'); // Ensure prompt is on a new line after final output
    }
}

/**
 * Schedule a UI redraw if one is not already scheduled
 */
export function scheduleRedraw() {
    if (!redrawScheduled) {
        setRedrawScheduled(true);
        setImmediate(redraw);
    }
}
