import { TestCaseAgent } from "@/agent"
import { Action, LLMClient, LLMClientIdentifier, ModelUsage } from "magnitude-core"
import EventEmitter from "eventemitter3";

export interface ActionDescriptor {
    action: Action,
    pretty: string
}

export interface StepDescriptor {
    variant: 'step',
    description: string,
    actions: ActionDescriptor[],
    //actions: Action[]
    status: 'pending' | 'running' | 'passed' | 'failed' | 'cancelled',
    thoughts?: string[]
}

export interface CheckDescriptor {
    variant: 'check',
    description: string,
    status: 'pending' | 'running' | 'passed' | 'failed' | 'cancelled'
}

// Used to pass up to UI and render stuff, for logging, etc.
export type TestStatus = 'pending' | 'running' | 'passed' | 'failed' | 'cancelled';

export interface TestState {
    status: TestStatus; // Added status property
    startedAt?: number,
	doneAt?: number,
    //cached?: boolean,
    stepsAndChecks: (StepDescriptor | CheckDescriptor)[],
    //actionCount: number,
    modelUsage: {
        llm: LLMClientIdentifier,
        inputTokens: number,
        outputTokens: number,
        numCalls: number
    }[],
	// macroUsage: {
	// 	provider: string,
	// 	model: string,
	// 	inputTokens: number,
	// 	outputTokens: number,
	// 	numCalls: number
	// }
	// microUsage: {
	// 	provider: string,
	// 	numCalls: number
	// },
    // Not known here since throw higher up
    failure?: TestFailure
}

export type TestResult = { 
    passed: true; 
    state?: TestState 
}
  | { 
    passed: false; 
    failure: TestFailure; 
    state?: TestState 
};

export interface TestFailure {
    message: string
}

export interface TestCaseEvents {
    'stateChanged': (state: TestState) => void
}


export class TestStateTracker {
    /**
     * Watches agent events and uses that to construct and update a test state
     */
    // Handles a lot of state tracking but not result/failure/done because the agent itself
    // doesn't know when done, that's known when the test fn finishes.
    private agent: TestCaseAgent;
    private state: TestState;
    // ref to last step, handy for attaching actions to it
    // private lastStep: StepDescriptor | null = null;
    // private lastCheck: CheckDescriptor | null = null;

    private lastStepOrCheck: StepDescriptor | CheckDescriptor | null = null;
    public readonly events: EventEmitter<TestCaseEvents>;

    constructor(agent: TestCaseAgent) {
        this.agent = agent;
        this.state = {
            status: 'pending', // Initialize status
            stepsAndChecks: [],
            modelUsage: [],
            // macroUsage: { provider: 'example', model: 'example', inputTokens: 0, outputTokens: 0, numCalls: 0 }, //agent.getMacro().getInfo(),
            // microUsage: { provider: 'example', numCalls: 0 } //agent.getMicro().getInfo()
        }
        this.agent.events.on('start', this.onStart, this);
        this.agent.events.on('stop', this.onStop, this);

        this.agent.events.on('actStarted', this.onActStarted, this);
        this.agent.events.on('actDone', this.onActDone, this);

        this.agent.events.on('actionStarted', this.onActionStarted, this);
        this.agent.events.on('actionDone', this.onActionDone, this);

        this.agent.events.on('tokensUsed', this.onTokensUsed, this);

        this.agent.checkEvents.on('checkStarted', this.onCheckStarted, this);
        this.agent.checkEvents.on('checkDone', this.onCheckDone, this);

        this.agent.events.on('thought', this.onThought, this);
        

        // this.agent.events.on('action', this.onAction, this);
        // this.agent.events.on('stepStart', this.onStepStart, this);
        // this.agent.events.on('checkStart', this.onCheckStart, this);
        // this.agent.events.on('stepSuccess', this.onStepSuccess, this);
        // this.agent.events.on('checkSuccess', this.onCheckSuccess, this);
        // this.agent.events.on('fail', this.onFail, this);
        
        // For forwarding state updates
        this.events = new EventEmitter();
    }

    // getEvents() {
    //     return this.events;
    // }

    getState() {
        return this.state;
    }

    // propagateState() {
    //     this.events.emit('update', this.state);
    // }

    onStart() {
        this.state.startedAt = Date.now();
        this.state.status = 'running'; // Set status to running
        this.events.emit('stateChanged', this.state);
    }

    onStop() {
        this.state.doneAt = Date.now();
        this.events.emit('stateChanged', this.state);
    }

    onTokensUsed(modelUsage: ModelUsage) {
        const modelHash = JSON.stringify(modelUsage.llm);
        let exists = false;
        for (const usage of this.state.modelUsage) {
            const compare = JSON.stringify(usage.llm);
            if (modelHash === compare) {
                // merge with existing usage
                exists = true;
                usage.inputTokens += modelUsage.inputTokens;
                usage.outputTokens += modelUsage.outputTokens;
                usage.numCalls += 1;
            }
        }
        if (!exists) {
            this.state.modelUsage.push({ ...modelUsage, numCalls: 1 })
        }
        // console.log('usage:', modelUsage);
        // console.log('total:', this.state.modelUsage);
    }

    onActionStarted(action: Action) {
        // TODO: maybe allow detatched actions (e.g. synthetic load at beginning, or manual low-level actions)
        if (!this.lastStepOrCheck || this.lastStepOrCheck.variant !== 'step') {
            throw new Error('Action reported without preceding step');
        }
        this.lastStepOrCheck.actions.push({
            action: action,
            pretty: this.agent.identifyAction(action).render(action)
        });
        this.events.emit('stateChanged', this.state);
    }

    onActionDone(action: Action) {
        // TODO: maybe allow detatched actions (e.g. synthetic load at beginning, or manual low-level actions)
        // if (!this.lastStepOrCheck || this.lastStepOrCheck.variant !== 'step') {
        //     throw new Error('Action reported without preceding step');
        // }
        // this.lastStepOrCheck.actions.push(action);
        // this.events.emit('update', this.state);
    }

    // onAction(action: ActionDescriptor) {
    //     // TODO: maybe allow detatched actions (e.g. synthetic load at beginning, or manual low-level actions)
    //     if (!this.lastStepOrCheck || this.lastStepOrCheck.variant !== 'step') {
    //         throw new Error('Action reported without preceding step');
    //     }
    //     this.lastStepOrCheck.actions.push(action);
    //     this.events.emit('update', this.state);
    // }

    onActStarted(task: string) {
        const stepDescriptor: StepDescriptor = {
            variant: 'step',
            description: task,
            actions: [],
            status: 'running'
        };
        this.state.stepsAndChecks.push(stepDescriptor);
        this.lastStepOrCheck = stepDescriptor;
        this.events.emit('stateChanged', this.state);
    }

    onActDone(task: string) {
        if (!this.lastStepOrCheck || this.lastStepOrCheck.variant !== 'step') {
            throw new Error('Step success without preceding step');
        }
        this.lastStepOrCheck.status = 'passed';
        // Note: Overall test status is not 'passed' yet, only this step.
        // TestRunner will set final test status.
        // Update any LLM metrics
        // this.state.macroUsage = this.agent.getMacro().getInfo();
        // this.state.microUsage = this.agent.getMicro().getInfo();
        this.events.emit('stateChanged', this.state);
    }

    onThought(thought: string) {
        if (this.lastStepOrCheck && this.lastStepOrCheck.variant === 'step') {
            if (!this.lastStepOrCheck.thoughts) this.lastStepOrCheck.thoughts = [];
            this.lastStepOrCheck.thoughts.push(thought);
        }
    }

    // onStepStart(description: string) {
    //     const stepDescriptor: StepDescriptor = {
    //         variant: 'step',
    //         description: description,
    //         actions: [],
    //         status: 'running'
    //     };
    //     this.state.stepsAndChecks.push(stepDescriptor);
    //     this.lastStepOrCheck = stepDescriptor;
    //     this.events.emit('update', this.state);
    // }

    // onStepSuccess() {
    //     if (!this.lastStepOrCheck || this.lastStepOrCheck.variant !== 'step') {
    //         throw new Error('Step success without preceding step');
    //     }
    //     this.lastStepOrCheck.status = 'passed';
    //     // Update any LLM metrics
    //     this.state.macroUsage = this.agent.getMacro().getInfo();
    //     this.state.microUsage = this.agent.getMicro().getInfo();
    //     this.events.emit('update', this.state);
    // }

    onCheckStarted(check: string) {
        const checkDescriptor: CheckDescriptor = {
            variant: 'check',
            description: check,
            status: 'running'
        };
        this.state.stepsAndChecks.push(checkDescriptor);
        this.lastStepOrCheck = checkDescriptor;
        this.events.emit('stateChanged', this.state);
    }

    onCheckDone(check: string, passed: boolean) {
        if (!this.lastStepOrCheck || this.lastStepOrCheck.variant !== 'check') {
            throw new Error('Check success reported without preceding check');
        }
        this.lastStepOrCheck.status = passed ? 'passed' : 'failed'; // Reflect check outcome
        // Note: Overall test status is not 'passed/failed' yet, only this check.
        // TestRunner will set final test status.
        // Update any LLM metrics
        // this.state.macroUsage = this.agent.getMacro().getInfo();
        // this.state.microUsage = this.agent.getMicro().getInfo();
        this.events.emit('stateChanged', this.state);
    }

    // onCheckStart(description: string) {
    //     const checkDescriptor: CheckDescriptor = {
    //         variant: 'check',
    //         description: description,
    //         status: 'running'
    //     };
    //     this.state.stepsAndChecks.push(checkDescriptor);
    //     this.lastStepOrCheck = checkDescriptor;
    //     this.events.emit('update', this.state);
    // }

    // onCheckSuccess() {
    //     if (!this.lastStepOrCheck || this.lastStepOrCheck.variant !== 'check') {
    //         throw new Error('Check success reported without preceding check');
    //     }
    //     this.lastStepOrCheck.status = 'passed';
    //     // Update any LLM metrics
    //     this.state.macroUsage = this.agent.getMacro().getInfo();
    //     this.state.microUsage = this.agent.getMicro().getInfo();
    //     this.events.emit('update', this.state);
    // }

    // onFail(failure: FailureDescriptor) {
    //     // if (!this.lastStepOrCheck) {
    //     //     throw new Error('Failure reported without preceding step or check');
    //     // }
    //     if (this.lastStepOrCheck) {
    //         if (failure.variant === 'cancelled') {
    //             this.lastStepOrCheck.status = 'cancelled';
    //         } else {
    //             this.lastStepOrCheck.status = 'failed';
    //         }
    //     }
    //     this.state.failure = failure;
    //     this.events.emit('update', this.state);
    // }
}
