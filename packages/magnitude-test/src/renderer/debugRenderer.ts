import { RegisteredTest } from "@/discovery/types";
import { TestState } from "@/runner/state";
import { TestRenderer } from "@/renderer";
import logger from '@/logger';

function statusFromState(state: TestState): 'pending' | 'running' | 'passed' | 'failed' {
    return state.failure
        ? 'failed'
        : (state.doneAt ? 'passed' : (state.startedAt ? 'running' : 'pending'));
}

function testToMeta(test: RegisteredTest): { id: string; title?: string; file?: string } {
    return {
        id: test.id,
        title: test.title,
        file: test.filepath
    };
}

function buildTestContext(state: TestState): {
    steps: number;
    checks: number;
    startedAt?: number;
    doneAt?: number;
    elapsedMs?: number;
    context: {
        step?: string;
        action?: string;
        stepStatus?: string;
        check?: string;
        checkStatus?: string;
        obs?: string;
    };
} {
    const now = Date.now();
    const startedAt = state.startedAt ?? null;
    const doneAt = state.doneAt ?? null;
    const elapsedMs = startedAt ? (doneAt ? doneAt - startedAt : now - startedAt) : null;

    let steps = 0;
    let checks = 0;
    let ctxParts: { step?: string; action?: string; stepStatus?: string; check?: string; checkStatus?: string; obs?: string } = {};

    if (Array.isArray(state.stepsAndChecks)) {
        for (const item of state.stepsAndChecks) {
            if (item.variant === 'check') checks++;
            else steps++;
        }

        const latest = state.stepsAndChecks[state.stepsAndChecks.length - 1];
        if (latest) {
            if (latest.variant === 'step') {
                if (latest.description) {
                    ctxParts.step = latest.description;
                }
                if (Array.isArray(latest.actions) && latest.actions.length > 0) {
                    const lastAction = latest.actions[latest.actions.length - 1];
                    if (lastAction?.pretty) {
                        ctxParts.action = lastAction.pretty;
                    }
                }
                if (latest.status === 'failed') {
                    ctxParts.stepStatus = 'failed';
                }
            } else {
                if (latest.description) {
                    ctxParts.check = latest.description;
                    if (latest.status === 'running') {
                        ctxParts.obs = `Verifying: ${latest.description}`;
                    } else if (latest.status === 'passed' || latest.status === 'failed') {
                        ctxParts.checkStatus = latest.status;
                    }
                }
            }
        }
    }

    return {
        steps,
        checks,
        startedAt: startedAt ?? undefined,
        doneAt: doneAt ?? undefined,
        elapsedMs: elapsedMs ?? undefined,
        context: ctxParts
    };
}

export class DebugRenderer implements TestRenderer {
    public stop() {
        return new Promise<void>((resolve, reject) => {
            logger.flush((err) => {
                if (err) reject(err);
                resolve();
            });
        });
    }

    public onTestStateUpdated(test: RegisteredTest, state: TestState): void {
        const status = statusFromState(state);
        const testMeta = testToMeta(test);
        const { steps, checks, startedAt, doneAt, elapsedMs, context } = buildTestContext(state);

        const payload = {
            test: testMeta,
            state: {
                status,
                startedAt,
                doneAt,
                elapsedMs
            },
            progress: { steps, checks },
            context,
            err: state.failure || undefined
        };

        if (status === 'failed') {
            logger.error(payload, 'test state');
        } else {
            logger.info(payload, 'test state');
        }
    }
}
