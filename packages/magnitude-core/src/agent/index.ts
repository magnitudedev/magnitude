import logger from '@/logger';
import EventEmitter from "eventemitter3";
import z from "zod";

import { Action } from "@/actions/types";
import { ModelHarness } from "@/ai/modelHarness";
import { AgentEvents } from "@/common/events";
import { AgentConnector } from '@/connectors';
import { Observation, RenderableContent } from '@/memory/observation';
import { LLMClient } from "@/ai/types";
import { AgentError } from "@/agent/errors";
import { AgentMemory, AgentMemoryOptions } from "@/memory";
import { ActionDefinition } from "@/actions";
import { taskActions } from "@/actions/taskActions";
import { ConnectorInstructions, AgentContext, traceAsync, MultiMediaContentPart } from "@/ai/baml_client";
import { telemetrifyAgent } from '@/telemetry/events';
import { isClaude } from '@/ai/util';
import { retryOnError } from '@/common';
import { renderContentParts } from '@/memory/rendering';
import { MultiModelHarness } from '@/ai/multiModelHarness';

const DEFAULT_MAX_STEPS = 100;

export interface AgentOptions {
    llm?: LLMClient | LLMClient[];
    connectors?: AgentConnector[];
    actions?: ActionDefinition<any>[]; // any additional actions not provided by connectors
    prompt?: string | null; // additional agent-level system prompt instructions
    telemetry?: boolean;
    //executor?: GroundingClient;
}

export interface ActOptions {
    prompt?: string // additional task-level system prompt instructions
    // TODO: reimpl, or maybe for tc agent specifically
    data?: RenderableContent //string | Record<string, string>
    memory?: AgentMemory // optional memory starting point
    maxSteps?: number; // Maximum number of steps for this act() call (default: 100)
    reuseMemory?: boolean; // Reuse memory from previous act() calls within the same agent instance
}

// Options for the startAgent helper function

const DEFAULT_CONFIG: Required<Omit<AgentOptions, 'actions'> & { actions: ActionDefinition<any>[] }> = {
    actions: [...taskActions], // Default to taskActions; other actions come from connectors
    connectors: [],
    llm: {
        provider: 'google-ai',
        options: {
            model: 'gemini-2.5-pro-preview-05-06',
            apiKey: process.env.GOOGLE_API_KEY || "YOUR_GOOGLE_API_KEY"
        }
    } as LLMClient,
    prompt: null,
    telemetry: true,
};

export class Agent {
    // maybe remove conns/actions from options since stored sep
    private options: Required<AgentOptions>//Omit<Required<AgentOptions>, 'actions'>;
    private connectors: AgentConnector[];
    private actions: ActionDefinition<any>[]; // actions from connectors + any other additional ones configured

    private memoryOptions: AgentMemoryOptions;

    public readonly models: MultiModelHarness;

    //public readonly model: ModelHarness;
    //public readonly micro: GroundingService;
    //public readonly events: EventEmitter<AgentEvents>;

    //protected readonly _emitter: EventEmitter<AgentEvents>;
    public readonly events: EventEmitter<AgentEvents> = new EventEmitter();
    
    //public readonly memory: AgentMemory;
    private doneActing: boolean;

    protected latestTaskMemory: AgentMemory;
    private persistentMemory?: AgentMemory; // Memory that persists across act() calls when reuseMemory is true

    constructor(baseConfig: Partial<AgentOptions> = {}) {
        this.options = {
            ...DEFAULT_CONFIG,
            ...baseConfig,
            connectors: baseConfig.connectors ?? [],
            actions: [...(baseConfig.actions || DEFAULT_CONFIG.actions)], 
        } as Required<AgentOptions>;

        this.connectors = this.options.connectors;

        // Aggregate actions from connectors
        //const aggregatedActions = [...this.options.actions];
        this.actions = [...this.options.actions];
        for (const connector of this.connectors) {
            this.actions.push(...(connector.getActionSpace ? connector.getActionSpace() : []));
        }
        // Deduplicate actions by name
        // TODO: maybe error instead, or automatically differentiate them?
        //this.options.actions = Array.from(new Map(aggregatedActions.map(actDef => [actDef.name, actDef])).values());

        const llms = Array.isArray(this.options.llm) ? this.options.llm : [this.options.llm];

        let doPromptCaching = false;
        for (const client of llms ) {
            // If any LLM is prompt-caching compatible, turn on prompt caching overall for memory etc.
            if (isClaude(client) && (client.provider === 'anthropic' || client.provider === 'claude-code')) {
                // Prompt-caching compatible client

                if ('promptCaching' in client.options && client.options.promptCaching !== undefined) {
                    doPromptCaching = client.options.promptCaching;
                } else {
                    // Default to true if not specified, and override on client config to true
                    doPromptCaching = true;
                    client.options.promptCaching = true;
                }
            }
        }

        //this.model = new ModelHarness({ llm: this.options.llm });
        this.models = new MultiModelHarness(llms);
        this.models.events.on('tokensUsed', (usage: any) => this.events.emit('tokensUsed', usage), this);
        this.doneActing = false;

        this.memoryOptions = {
            // TODO: maybe do if Gemini or other prompt caching supported providers as well
            // Claude supports prompt caching but only via Anthropic, not on Bedrock
            promptCaching: doPromptCaching
        };

        // Empty memory will get replaced on first act(), but this prevents errors from having undefined memory
        this.latestTaskMemory = new AgentMemory(this.memoryOptions);
    }

    public getConnector<C extends AgentConnector>(
        connectorClass: new (...args: any[]) => C
    ): C | undefined {
        return this.connectors.find(c => c instanceof connectorClass) as C | undefined;
    }

    public require<C extends AgentConnector>(
        connectorClass: new (...args: any[]) => C
    ): C {
        const connector = this.getConnector(connectorClass);
        if (!connector) throw new Error(`Missing required connector ${connectorClass}`);
        return connector;
    }

    async start(): Promise<void> { 
        // Register telemetry if enabled - do on start instead of cons to prevent weird subclass event issues
        if (this.options.telemetry) telemetrifyAgent(this);

        //console.log('setting up model')
        await this.models.setup();
        //console.log('done setting up model')

        logger.info("Agent: Starting connectors...");
        for (const connector of this.connectors) {
            if (connector.onStart) await connector.onStart(); 
        }
        this.events.emit('start');
        logger.info("Agent: All connectors started.");

        // logger.info("Making initial observations...");
        // await this._recordConnectorObservations();
        // logger.info("Initial observations recorded");
        // Initial observations are handled by the first getObservations call in exec
    }

    identifyAction(action: Action) {
        // Get definition corresponding to an action
        const actionDefinition = this.actions.find(def => def.name === action.variant);

        if (!actionDefinition) {
            // It's possible the action name was from a connector that is no longer active,
            // or the action space was not correctly aggregated.
            throw new AgentError(`Undefined action type '${action.variant}'. Ensure agent is configured with appropriate action definitions from connectors.`);
        }
        return actionDefinition;
    }
    
    async exec(action: Action, memory?: AgentMemory): Promise<void> {
        /**
         * Execute an action that belongs to this Agent's action space.
         * Provide memory to record the action taken, its results, and any connector observations to that memory.
         */
        let actionDefinition = this.identifyAction(action);
        
        let input: any;
        if (actionDefinition.schema instanceof z.ZodObject) {
            let variant: string;
            ({ variant, ...input } = action);
        } else {
            input = (action as any).input; 
        }

        let parsed = actionDefinition.schema.safeParse(input);

        if (!parsed.success) {
            throw new AgentError(`Generated action '${action.variant}' violates input schema: ${parsed.error.message}`, { adaptable: true });
        }

        this.events.emit('actionStarted', action);
        
        const data = await actionDefinition.resolver(
            { input: parsed.data, agent: this }
        );

        this.events.emit('actionDone', action);

        if (memory) {
            // Record action taken
            memory.recordObservation(Observation.fromActionTaken(actionDefinition.name, JSON.stringify(action)));

            // Record results of action
            if (data) {
                memory.recordObservation(Observation.fromActionResult(actionDefinition.name, data));
            }

            // Collect and record observations from connectors
            await this._recordConnectorObservations(memory);
        }
    }

    protected async _recordConnectorObservations(memory: AgentMemory) {
        for (const connector of this.connectors) {
            // could do Promise.all if matters
            const connObservations = connector.collectObservations ? await connector.collectObservations() : [];
            //observations.push(...connObservations);
            for (const obs of connObservations) {
                memory.recordObservation(obs);
            }
        }
    }

    get memory(): AgentMemory {
        //if (!this.latestTaskMemory) throw new Error("No memory available");
        return this.latestTaskMemory;
    }

    async act(taskOrSteps: string | string[], options: ActOptions = {}): Promise<void> {
        const instructions = [
            ...(this.options.prompt ? [this.options.prompt] : []),
            ...(options.prompt ? [options.prompt] : []),
        ].join('\n');
        
        let taskMemory: AgentMemory;
        
        // First priority: use provided memory if available
        if (options.memory) {
            taskMemory = options.memory;
            // Optionally save for future reuse if requested
            if (options.reuseMemory) {
                this.persistentMemory = taskMemory;
                logger.debug('Using provided memory and saving for future reuse');
            }
        } 
        // Second priority: reuse persistent memory if requested
        else if (options.reuseMemory && this.persistentMemory) {
            // Reuse existing memory for this execution
            taskMemory = this.persistentMemory;
            logger.debug('Reusing persistent memory from previous act() calls');
        } 
        // Default: create new memory
        else {
            // Create new memory
            taskMemory = new AgentMemory({ 
                ...this.memoryOptions, 
                instructions: instructions === '' ? undefined : instructions 
            });
            
            // Save for future reuse if requested
            if (options.reuseMemory) {
                this.persistentMemory = taskMemory;
                logger.debug('Created new persistent memory for future reuse');
            }
        }

        if (Array.isArray(taskOrSteps)) {
            const steps = taskOrSteps;

            //this.events.emit('actStarted', steps.join(', '));

            // trace overall task
            await (traceAsync('multistep', async (steps: string[], options: ActOptions) => {
                for (const step of steps) {
                    this.events.emit('actStarted', step, options);
                    await this._traceAct(step, taskMemory, options);
                    this.events.emit('actDone', step, options);
                }
            })(steps, options));

            //this.events.emit('actDone', steps.join(', '));
        } else {
            const task = taskOrSteps;

            this.events.emit('actStarted', task, options);

            await this._traceAct(task, taskMemory, options);
            this.events.emit('actDone', task, options);
        }
    }

    async _traceAct(task: string, memory: AgentMemory, options: ActOptions = {}) {
        // memory not serializable to trace so bake it
        await (traceAsync('act', async (task: string) => {
            await this._act(task, memory, options);
        })(task));
    }

    private async _buildContext(memory: AgentMemory): Promise<AgentContext> {
        const messages = await memory.render();

        const connectorInstructions: ConnectorInstructions[] = [];

        for (const connector of this.connectors) {
            if (connector.getInstructions) {
                const instructions = await connector.getInstructions();

                if (instructions) {
                    connectorInstructions.push({
                        connectorId: connector.id,
                        instructions: instructions
                    });
                }
            }
        }

        return {
            instructions: memory.instructions,
            observationContent: messages,
            //observationContent: content,
            connectorInstructions: connectorInstructions
        };
    }

    async _act(description: string, memory: AgentMemory, options: ActOptions = {}): Promise<void> {
        this.doneActing = false;
        const maxSteps = options.maxSteps ?? DEFAULT_MAX_STEPS; // Default to 100 steps
        let currentStep = 0;
        
        logger.info(`Act: ${description} (max steps: ${maxSteps})`);

        // for now simply add data to task
        let dataContentParts: MultiMediaContentPart[] = [];
        if (options.data) {
            //description += "\nUse the following data where appropriate:\n";
            // description += "\n<data>\n";
            // // if (typeof options.data === 'string') {
            // //     description += options.data;
            // // } else {
            // //     description += Object.entries(options.data).map(([k, v]) => `${k}: ${v}`).join("\n");
            // // }
            // const parts = renderParts(options.data);
            // description += "\n</data>";
            dataContentParts = await renderContentParts(options.data, { mode: 'json', indent: 2 });
        }
        //this.events.emit('stepStart', description);

        //const testData = convertOptionsToTestData(options);

        // Initialize task memory and record initial observations
        // Combine any agent-level and task-level instructions
        
        this.latestTaskMemory = memory;

        // record initial observations
        logger.info("Making initial observations...");
        await this._recordConnectorObservations(memory);
        logger.info("Initial observations recorded");

        while (!this.doneActing && currentStep < maxSteps) {
            // Removed direct screenshot/tabState access here; it's part of memoryContext via connectors
            logger.info(`Creating partial recipe (step ${currentStep + 1}/${maxSteps})`);

            let reasoning: string = "";
            let actions: Action[] = [];

            try {
                const memoryContext = await this._buildContext(memory);
                await retryOnError(
                    async () => {
                        ({ reasoning, actions } = await this.models.partialAct(
                            memoryContext,
                            description,
                            dataContentParts,
                            this.actions 
                        ));
                        if (actions.length === 0) {
                            // Empty action list behavior - default wait else ... err? what if not in action space?
                            //actions.push()
                            throw new AgentError(`No actions generated`);
                        }
                    },
                    // HTTP body is not JSON - comes from Anthropic sometimes, weird error
                    // Sometimes Anthropic will give 401 Unauthorized randomly even when authorized
                    {
                        mode: 'retry_on_partial_message',
                        errorSubstrings: ['HTTP body is not JSON', '401 Unauthorized', 'No actions generated'],
                        retryLimit: 3,
                        delayMs: 1000,
                        showWarnOnRetry: true
                    }
                );
            } catch (error: unknown) {
                logger.error(`Error planning actions: ${error instanceof Error ? error.message : String(error)}`);
                /**
                 * (1) Failure to conform to JSON
                 * (2) Misconfigured BAML client / bad API key
                 * (3) Network error (past max retries)
                 */
                // this.fail({
                //     variant: 'misalignment',
                //     message: `Could not create partial recipe -> ${(error as Error).message}`
                // });
                throw new AgentError(
                    `Error planning actions: ${(error as Error).message}`, { variant: 'misalignment' }
                )
            }

            logger.info({ reasoning, actions }, `Partial recipe created`);
            
            // Could be emitted in memory and bubbled up instead of recordThought was called in more places
            this.events.emit('thought', reasoning);
            memory.recordThought(reasoning);

            // Execute partial recipe
            for (const action of actions) {
                await this.exec(action, memory);

                // const postActionScreenshot = await this.screenshot();
                // const actionDescriptor: ActionDescriptor = { ...action, screenshot: postActionScreenshot.image } as ActionDescriptor;
                // this.events.emit('action', actionDescriptor);
                logger.info({ action }, `Action taken`);
            }

            // If macro expects these actions should complete the step, break
            // if (finished) {
            //     break;
            // }
            if (this.doneActing) {
                break;
            }
            
            currentStep++;
        }
        
        if (currentStep >= maxSteps && !this.doneActing) {
            logger.warn(`Reached maximum steps limit (${maxSteps}) without completing task: ${description}`);
            this.events.emit('maxStepsReached', description, maxSteps);
        }

        logger.info(`Done with step after ${currentStep} steps`);
        //this.events.emit('stepSuccess');
        //this.currentTaskMemory = null;
    }

    async query<T extends z.Schema>(query: string, schema: T): Promise<z.infer<T>> {
        // Record observations in case no act() was used beforehand
        await this._recordConnectorObservations(this.latestTaskMemory);
        const memoryContext = await this._buildContext(this.memory);//this.memory.buildContext(this.connectors);
        return await this.models.query(memoryContext, query, schema);
    }

    async queueDone() {
        this.doneActing = true;
    }

    async stop() {
        /**
         * Stop the agent and close the browser context.
         * May be called asynchronously and interrupt an agent in the middle of a action sequence.
         */
        // set signal to cancelled?
        logger.info("Agent: Stopping connectors...");
        for (const connector of this.connectors) {
            try {
                if (connector.onStop) await connector.onStop();
            } catch (error) {
                logger.warn(`Agent: Error stopping connector ${connector.id}: ${error instanceof Error ? error.message : String(error)}`);
            }
        }
        this.events.emit('stop');
        logger.info("Agent: All connectors stopped.");
        logger.info("Agent: Stopped successfully.");
    }

    // async dumpMemoryJSON() {
    //     return await this.memory.toJSON();
    // }
}
