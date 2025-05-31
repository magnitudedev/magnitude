import logUpdate from 'log-update';
import { CategorizedTestCases, TestRunnable } from '@/discovery/types';
import { TestState, AllTestStates } from './types';
import { FailureDescriptor, variantToTitle } from 'magnitude-core';
import { VERSION } from '@/version';
import { formatDuration, getUniqueTestId, wrapText } from './util';
import { 
    ANSI_RESET, ANSI_GREEN, ANSI_BRIGHT_BLUE, 
    ANSI_GRAY, ANSI_RED, ANSI_BOLD, ANSI_DIM, PADDING, BOX_CHARS_ROUNDED
} from './constants';
import { 
    str, createBoxAnsi, insertLineIntoBoxAnsi, styleAnsi, 
    describeAction, getActionSymbol, getTestStatusIndicatorChar, 
    getStepStatusIndicatorChar, getCheckStatusIndicatorChar 
} from './drawingUtils';
import {
    currentWidth, redrawScheduled, currentTestStates, currentTests,
    currentModel, elapsedTimes, isFinished, spinnerFrame,
    lastOutputLineCount, isFirstDraw, isResizing, renderSettings, // Added renderSettings
    setRedrawScheduled, setLastOutputLineCount, setIsFirstDraw
} from './uiState';
import { spinnerChars } from './constants';
import { knownCostMap } from '@/util';

/**
 * Generate the title bar portion of the UI
 * @returns Array of strings with ANSI codes representing the title bar
 */
export function generateTitleBarString(): string[] {
    // Returns array of strings with ANSI codes
    const boxLines = createBoxAnsi(currentWidth, 3, ANSI_BRIGHT_BLUE);
    const titleText = `${ANSI_BRIGHT_BLUE}${ANSI_BOLD}Magnitude v${VERSION}${ANSI_RESET}`;
    const modelText = `${ANSI_GRAY}${currentModel}${ANSI_RESET}`;
    const contentWidth = currentWidth - 2; // Width inside vertical bars

    // Construct the middle line directly
    const titleWidth = str(titleText);
    const modelWidth = str(modelText);
    const singleSpacePadding = 1;
    const spaceBetween = contentWidth - titleWidth - modelWidth - (singleSpacePadding * 2); // Account for 1 space on each side
    const middleLineContent = ' '.repeat(singleSpacePadding) + titleText + ' '.repeat(Math.max(0, spaceBetween)) + modelText + ' '.repeat(singleSpacePadding); // Add 1 space padding

    // Replace the default middle line (index 1)
    boxLines[1] = `${ANSI_BRIGHT_BLUE}${BOX_CHARS_ROUNDED.vertical}${middleLineContent.padEnd(contentWidth)}${BOX_CHARS_ROUNDED.vertical}${ANSI_RESET}`;

    return boxLines;
}

/**
 * Generate a string representation of a failure
 * @param failure The failure descriptor
 * @param indent Indentation level
 * @param availableWidth Available width for content
 * @returns Array of strings with ANSI codes
 */
export function generateFailureString(failure: FailureDescriptor, indent: number, availableWidth: number): string[] {
    // Returns array of strings with ANSI codes
    const output: string[] = [];
    const prefix = '↳ ';
    const prefixAnsi = `${ANSI_RED}${prefix}${ANSI_RESET}`;
    const contentWidth = Math.max(1, availableWidth - str(prefix));

    const addLine = (text: string, styleCode = ANSI_RED, bold = false) => {
        const fullStyleCode = `${styleCode}${bold ? ANSI_BOLD : ''}`;
        wrapText(text, contentWidth).forEach((line, index) => {
            const linePrefix = index === 0 ? prefixAnsi : ' '.repeat(str(prefix));
            // Ensure reset at the end of the styled line part
            output.push(' '.repeat(indent) + linePrefix + `${fullStyleCode}${line}${ANSI_RESET}`);
        });
    };

    const addSimpleLine = (text: string, styleCode = ANSI_RED) => {
        output.push(' '.repeat(indent) + prefixAnsi + `${styleCode}${text}${ANSI_RESET}`);
    };

    if (failure.variant === 'bug') {
        addLine(`Found bug: ${failure.title}`, ANSI_RED, true); // Bold Red
        addLine(`Expected: ${failure.expectedResult}`);
        addLine(`Actual:   ${failure.actualResult}`);
        addSimpleLine(`Severity: ${failure.severity.toUpperCase()}`);
    } else if (failure.variant === 'cancelled') {
        addSimpleLine('Cancelled', ANSI_GRAY);
    } else {
        const typedFailure = failure as Extract<FailureDescriptor, { message?: string }>;
        if ('message' in typedFailure && typedFailure.message) {
            addLine(`${variantToTitle(typedFailure.variant)}: ${typedFailure.message}`);
        } else {
            addSimpleLine(typedFailure.variant || 'unknown error');
        }
    }
    return output;
}

/**
 * Generate a string representation of a test
 * @param test The test runnable
 * @param state The test state
 * @param filepath The file path
 * @param groupName The group name (or null)
 * @param indent Indentation level
 * @param availableWidth Available width for content
 * @returns Array of strings with ANSI codes
 */
export function generateTestString(test: TestRunnable, state: TestState, filepath: string, groupName: string | null, indent: number, availableWidth: number): string[] {
    // Returns array of strings with ANSI codes
    const output: string[] = [];
    const testId = getUniqueTestId(filepath, groupName, test.title);
    const contentWidth = Math.max(1, availableWidth - indent);
    const stepIndent = indent + 2;
    const actionIndent = stepIndent + 2;
    const stepContentWidth = Math.max(1, availableWidth - stepIndent - 2);
    const actionContentWidth = Math.max(1, availableWidth - actionIndent - 2);

    // --- Test Title Line ---
    const statusCharPlain = state.status === 'running' ? spinnerChars[spinnerFrame] : getTestStatusIndicatorChar(state.status);
    const statusStyled = styleAnsi(state.status, statusCharPlain, 'test');

    const timerText = state.status !== 'pending' ? `${ANSI_GRAY} [${formatDuration(elapsedTimes[testId] ?? 0)}]${ANSI_RESET}` : '';
    const titleAvailableWidth = contentWidth - 2 - str(timerText); // Use str for width
    const wrappedTitle = wrapText(test.title, titleAvailableWidth > 10 ? titleAvailableWidth : contentWidth - 2);

    wrappedTitle.forEach((line, index) => {
        const linePrefix = index === 0 ? `${statusStyled} ` : '  ';
        const lineSuffix = index === 0 ? timerText : '';
        output.push(' '.repeat(indent) + linePrefix + line + lineSuffix); // Title is plain
    });

    // --- Steps and Checks ---
    if (state.stepsAndChecks.length > 0) {
        state.stepsAndChecks.forEach((item) => {
            const itemIndent = stepIndent;
            const itemContentWidth = stepContentWidth;
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
            const wrappedDesc = wrapText(itemDesc, itemContentWidth);

            wrappedDesc.forEach((line, index) => {
                const linePrefix = index === 0 ? `${styledChar} ` : '  ';
                output.push(' '.repeat(itemIndent) + linePrefix + line); // Desc is plain
            });

            // Draw actions only for steps if showActions is true
            if (renderSettings.showActions && item.variant === 'step' && item.actions.length > 0) {
                item.actions.forEach(action => {
                    const actionSymbol = `${ANSI_GRAY}${getActionSymbol(action.variant)}${ANSI_RESET}`;
                    const actionDesc = describeAction(action); // Plain desc
                    const wrappedActionDesc = wrapText(actionDesc, actionContentWidth);
                    wrappedActionDesc.forEach((line, index) => {
                        const linePrefix = index === 0 ? `${actionSymbol} ` : '  ';
                        output.push(' '.repeat(actionIndent) + linePrefix + `${ANSI_GRAY}${line}${ANSI_RESET}`); // Gray action text
                    });
                });
            }
        });
    }

    // --- Failure ---
    if (state.failure && state.failure.variant !== 'cancelled') {
        const failureLines = generateFailureString(state.failure, stepIndent, availableWidth - stepIndent);
        output.push(...failureLines);
    }

    return output;
}

/**
 * Generate the test list portion of the UI
 * @param boxHeight Height of the box
 * @returns Array of strings with ANSI codes representing the test list
 */
export function generateTestListString(boxHeight: number): string[] {
    // Returns array of strings with ANSI codes
    const boxLines = createBoxAnsi(currentWidth, boxHeight, ANSI_GRAY); // Gray box
    const contentWidth = currentWidth - (PADDING * 2); // Content width inside padding
    const fileIndent = 0; // Relative to content area start (after padding)
    const groupIndent = fileIndent + 2;
    const testBaseIndent = groupIndent;

    let currentContentLine = 0; // Tracks lines *within* the box content area (0-based)
    const maxContentLines = boxHeight - 2;

    for (const [filepath, { ungrouped, groups }] of Object.entries(currentTests)) {
        if (currentContentLine >= maxContentLines) break;

        const fileHeader = `${ANSI_BRIGHT_BLUE}${ANSI_BOLD}☰ ${filepath}${ANSI_RESET}`;
        insertLineIntoBoxAnsi(boxLines, fileHeader, currentContentLine + 1, (PADDING - 1) + fileIndent, currentWidth);
        currentContentLine++;

        // Draw ungrouped tests
        if (ungrouped.length > 0) {
            for (const test of ungrouped) {
                if (currentContentLine >= maxContentLines) break;
                const testId = getUniqueTestId(filepath, null, test.title);
                const state = currentTestStates[testId];
                if (state) {
                    const testLines = generateTestString(test, state, filepath, null, testBaseIndent, contentWidth - testBaseIndent);
                    testLines.forEach(line => {
                        if (currentContentLine < maxContentLines) {
                            insertLineIntoBoxAnsi(boxLines, line, currentContentLine + 1, PADDING - 1, currentWidth);
                            currentContentLine++;
                        }
                    });
                }
            }
        }
        if (currentContentLine >= maxContentLines) break;

        // Draw grouped tests
        if (Object.entries(groups).length > 0) {
            for (const [groupName, groupTests] of Object.entries(groups)) {
                if (currentContentLine >= maxContentLines) break;
                const groupHeader = `${ANSI_BRIGHT_BLUE}${ANSI_BOLD}↳ ${groupName}${ANSI_RESET}`;
                insertLineIntoBoxAnsi(boxLines, groupHeader, currentContentLine + 1, (PADDING - 1) + groupIndent, currentWidth);
                currentContentLine++;

                for (const test of groupTests) {
                    if (currentContentLine >= maxContentLines) break;
                    const testId = getUniqueTestId(filepath, groupName, test.title);
                    const state = currentTestStates[testId];
                    if (state) {
                        const testLines = generateTestString(test, state, filepath, groupName, testBaseIndent + 2, contentWidth - (testBaseIndent + 2));
                        testLines.forEach(line => {
                            if (currentContentLine < maxContentLines) {
                                insertLineIntoBoxAnsi(boxLines, line, currentContentLine + 1, PADDING - 1, currentWidth);
                                currentContentLine++;
                            }
                        });
                    }
                }
                if (currentContentLine >= maxContentLines) break;
            }
        }

        // Add blank line between files if space allows
        if (currentContentLine < maxContentLines) {
            insertLineIntoBoxAnsi(boxLines, '', currentContentLine + 1, PADDING - 1, currentWidth);
            currentContentLine++;
        }
    }

    return boxLines;
}

/**
 * Generate the summary portion of the UI
 * @param boxHeight Height of the box
 * @returns Array of strings with ANSI codes representing the summary
 */
export function generateSummaryString(boxHeight: number): string[] {
    // Returns array of strings with ANSI codes
    let totalInputTokens = 0;
    let totalOutputTokens = 0;
    const statusCounts = { pending: 0, running: 0, passed: 0, failed: 0, cancelled: 0, total: 0 };
    const failuresWithContext: { filepath: string; groupName: string | null; testTitle: string; failure: FailureDescriptor }[] = [];
    const testContextMap = new Map<string, { filepath: string; groupName: string | null; testTitle: string }>();

    Object.entries(currentTests).forEach(([filepath, { ungrouped, groups }]) => {
        ungrouped.forEach(test => testContextMap.set(getUniqueTestId(filepath, null, test.title), { filepath, groupName: null, testTitle: test.title }));
        Object.entries(groups).forEach(([groupName, groupTests]) => groupTests.forEach(test => testContextMap.set(getUniqueTestId(filepath, groupName, test.title), { filepath, groupName, testTitle: test.title })));
    });

    Object.entries(currentTestStates).forEach(([testId, state]) => {
        statusCounts.total++;
        statusCounts[state.status]++;
        totalInputTokens += state.macroUsage.inputTokens;
        totalOutputTokens += state.macroUsage.outputTokens;
        if (state.failure && state.failure.variant !== 'cancelled') {
            const context = testContextMap.get(testId);
            failuresWithContext.push({
                filepath: context?.filepath ?? 'Unknown File', groupName: context?.groupName ?? null,
                testTitle: context?.testTitle ?? 'Unknown Test', failure: state.failure
            });
        }
    });

    const hasFailures = failuresWithContext.length > 0;
    const boxColor = hasFailures ? ANSI_RED : ANSI_GRAY;
    const boxLines = createBoxAnsi(currentWidth, boxHeight, boxColor);
    const baseContentWidth = currentWidth - (PADDING * 2); // Width inside padding (original contentWidth)
    const summaryInternalLeftPadding = 1;
    const effectiveSummaryContentWidth = baseContentWidth - summaryInternalLeftPadding; // Actual width for text after internal padding

    let currentContentLine = 0; // 0-based index for content lines
    const maxContentLines = boxHeight - 2;

    // --- Status Counts Line ---
    if (currentContentLine < maxContentLines) {
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
                costDescription = ` (\$${((totalInputTokens*inputCost + totalOutputTokens*outputCost) / 1000000).toFixed(2)})`;
            }
        }
        
        let tokenText = `${ANSI_GRAY}tokens: ${totalInputTokens} in, ${totalOutputTokens} out${costDescription}${ANSI_RESET}`;
        
        const spaceNeeded = str(statusLine) + str(tokenText);
        const spacer = ' '.repeat(Math.max(0, effectiveSummaryContentWidth - spaceNeeded)); // Use effective width
        const combinedLine = statusLine + spacer + tokenText;

        insertLineIntoBoxAnsi(boxLines, combinedLine, currentContentLine + 1, summaryInternalLeftPadding, currentWidth); // Apply internal padding
        currentContentLine++;
    }

    // --- Failures ---
    if (hasFailures && currentContentLine < maxContentLines) {
        const failureHeader = `${ANSI_DIM}Failures:${ANSI_RESET}`; // Dim
        insertLineIntoBoxAnsi(boxLines, failureHeader, currentContentLine + 1, summaryInternalLeftPadding, currentWidth); // Apply internal padding
        currentContentLine++;

        for (const { filepath, groupName, testTitle, failure } of failuresWithContext) {
            if (currentContentLine >= maxContentLines) break;
            const contextString = `${ANSI_DIM}${filepath}${groupName ? ` > ${groupName}` : ''} > ${testTitle}${ANSI_RESET}`; // Dim
            insertLineIntoBoxAnsi(boxLines, contextString, currentContentLine + 1, 1 + summaryInternalLeftPadding, currentWidth); // Indent context + internal padding
            currentContentLine++;

            if (currentContentLine < maxContentLines) {
                const failureLines = generateFailureString(failure, 2, effectiveSummaryContentWidth - 2); // Use effective width, indent failure details further
                failureLines.forEach(line => {
                    if (currentContentLine < maxContentLines) {
                        insertLineIntoBoxAnsi(boxLines, line, currentContentLine + 1, summaryInternalLeftPadding, currentWidth); // Apply internal padding
                        currentContentLine++;
                    }
                });
            }

            if (currentContentLine < maxContentLines) {
                insertLineIntoBoxAnsi(boxLines, '', currentContentLine + 1, summaryInternalLeftPadding, currentWidth); // Space line with internal padding
                currentContentLine++;
            }
        }
    }

    return boxLines;
}

/**
 * Calculate the height needed for the test list based on current tests and states
 * @param tests The categorized test cases
 * @param testStates The current test states
 * @returns The calculated height
 */
export function calculateTestListHeight(tests: CategorizedTestCases, testStates: AllTestStates): number {
    let height = 0;
    
    for (const [filepath, { ungrouped, groups }] of Object.entries(tests)) {
        height++; // File header line
        
        // Calculate height for ungrouped tests
        if (ungrouped.length > 0) {
            for (const test of ungrouped) {
                const testId = getUniqueTestId(filepath, null, test.title);
                const state = testStates[testId];
                if (state) {
                    // Calculate the content width for estimating line wrapping
                    const contentWidth = Math.max(1, currentWidth - (PADDING * 2) - 4);
                    
                    // Test title (at least 1 line)
                    height += wrapText(test.title, contentWidth - 2).length;
                    
                    // Steps and checks
                    state.stepsAndChecks.forEach(item => {
                        // Description lines
                        height += wrapText(item.description, contentWidth - 4).length;
                        
                        // Action lines for steps
                        if (renderSettings.showActions && item.variant === 'step' && item.actions.length > 0) {
                            item.actions.forEach(action => {
                                height += wrapText(describeAction(action), contentWidth - 6).length;
                            });
                        }
                    });
                    
                    // Failure lines
                    if (state.failure && state.failure.variant !== 'cancelled') {
                        // Add approximate lines for failure (simplified estimate)
                        height += 2; // Minimum lines for a failure
                        if (state.failure.variant === 'bug') {
                            height += 3; // Extra lines for bug details
                        }
                    }
                }
            }
        }
        
        // Calculate height for grouped tests
        if (Object.entries(groups).length > 0) {
            for (const [groupName, groupTests] of Object.entries(groups)) {
                height++; // Group header line
                
                for (const test of groupTests) {
                    const testId = getUniqueTestId(filepath, groupName, test.title);
                    const state = testStates[testId];
                    if (state) {
                        // Same calculation as above but with deeper indentation
                        const contentWidth = Math.max(1, currentWidth - (PADDING * 2) - 6);
                        
                        // Test title (at least 1 line)
                        height += wrapText(test.title, contentWidth - 2).length;
                        
                        // Steps and checks
                        state.stepsAndChecks.forEach(item => {
                            // Description lines
                            height += wrapText(item.description, contentWidth - 4).length;
                            
                            // Action lines for steps
                            if (renderSettings.showActions && item.variant === 'step' && item.actions.length > 0) {
                                item.actions.forEach(action => {
                                    height += wrapText(describeAction(action), contentWidth - 6).length;
                                });
                            }
                        });
                        
                        // Failure lines
                        if (state.failure && state.failure.variant !== 'cancelled') {
                            // Add approximate lines for failure (simplified estimate)
                            height += 2; // Minimum lines for a failure
                            if (state.failure.variant === 'bug') {
                                height += 3; // Extra lines for bug details
                            }
                        }
                    }
                }
            }
        }
        
        height++; // Blank line between files
    }
    
    return Math.max(3, height); // Ensure at least a minimum height for the box
}

/**
 * Calculate the height needed for the summary based on current test states
 * @param testStates The current test states
 * @returns The calculated height
 */
export function calculateSummaryHeight(testStates: AllTestStates): number {
    let height = 0;
    height++; // Status counts line

    const failuresWithContext: { failure: FailureDescriptor }[] = [];
    Object.entries(testStates).forEach(([_, state]) => {
        if (state.failure && state.failure.variant !== 'cancelled') {
            failuresWithContext.push({ failure: state.failure });
        }
    });

    if (failuresWithContext.length > 0) {
        height++; // "Failures:" title
        failuresWithContext.forEach(({ failure }) => {
            height++; // Context line
            const contentWidth = Math.max(1, currentWidth - (PADDING * 2) - 4); // Approx width for failure text
            // Estimate height based on plain text wrapping
            if (failure.variant === 'bug') {
                height += wrapText(`Found bug: ${failure.title}`, contentWidth).length;
                height += wrapText(`Expected: ${failure.expectedResult}`, contentWidth).length;
                height += wrapText(`Actual:   ${failure.actualResult}`, contentWidth).length;
                height += 1; // Severity line
            } else if ('message' in failure) {
                const typedFailure = failure as Extract<FailureDescriptor, { message?: string }>;
                const prefixMap: Partial<Record<FailureDescriptor['variant'], string>> = { /*...*/ }; // Keep this for potential future use
                const failurePrefix = prefixMap[typedFailure.variant] || `${typedFailure.variant}: `;
                height += wrapText(failurePrefix + (typedFailure.message || ''), contentWidth).length;
            } else {
                height += 1; // Fallback line for other failure types
            }
            height++; // Space after failure
        });
    }
    return height;
}

/**
 * Main function to redraw the UI
 */
export function redraw() {
    setRedrawScheduled(false);

    // --- Calculate Layout ---
    const testListMinHeight = 3; // Minimum 3 lines for test list box
    const summaryMinHeight = 3;  // Minimum 3 lines for summary box

    // Calculate the actual height needed for the test list
    let testListHeight = calculateTestListHeight(currentTests, currentTestStates);
    if (testListHeight > 0) { // If there's content for the test list
        testListHeight += 2; // Add 2 for box borders
        testListHeight = Math.max(testListMinHeight, testListHeight); // Ensure minimum height
    } else {
        testListHeight = 0; // No content, no box
    }

    // Calculate the actual height needed for the summary
    let summaryHeight = calculateSummaryHeight(currentTestStates);
    if (summaryHeight > 0) { // If there's content for the summary
        summaryHeight += 2; // Add 2 for box borders
        summaryHeight = Math.max(summaryMinHeight, summaryHeight); // Ensure minimum height
    } else {
        // If summary content is 0, but there are tests, we might still want to show an empty summary box
        // For now, let's keep it simple: if calculateSummaryHeight is 0, summaryHeight is 0.
        // This means an empty summary (e.g. no failures, 0 tokens) won't render a box.
        // If we want to always show the summary box if tests are running, this logic would need adjustment.
        summaryHeight = 0; 
    }
    
    // Spacing is not used between test list and summary if they stack vertically and grow.
    const spacingHeight = 0;

    // --- Generate Output Strings ---
    const outputLines: string[] = [];
    outputLines.push(''); // Add a sacrificial blank line at the very top
    outputLines.push(...generateTitleBarString());

    if (testListHeight > 0) {
        outputLines.push(...generateTestListString(testListHeight));
    }

    if (spacingHeight > 0) {
        outputLines.push(''); // Spacing only if both are present
    }

    if (summaryHeight > 0) {
        outputLines.push(...generateSummaryString(summaryHeight));
    }

    // --- Update Terminal using log-update ---
    const frameContent = outputLines.join('\n');

    // Only clear the screen in specific cases to avoid scrollback buffer issues
    // Don't clear during resize operations or if line count hasn't changed
    const shouldClear = !isFirstDraw && 
                        !isResizing && 
                        outputLines.length !== lastOutputLineCount;
                        
    // Clear only when necessary to avoid unnecessary screen clearing
    if (shouldClear) {
        logUpdate.clear();
    }

    // Always update with the latest content
    logUpdate(frameContent);
    setLastOutputLineCount(outputLines.length); // Store line count for next redraw

    if (isFirstDraw) {
        setIsFirstDraw(false);
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
