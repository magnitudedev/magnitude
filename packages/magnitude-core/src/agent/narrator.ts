import { Action } from '@/actions/types';
import { ActOptions, Agent } from '@/agent';
import { blueBright, bold, cyanBright, gray } from 'ansis';
import { BrowserAgent } from './browserAgent';
import { z } from 'zod';

export function narrateAgent(agent: Agent) {
    let totalInputTokens = 0;
    let totalOutputTokens = 0;
    let totalInputTokenCost = 0.0;
    let totalOutputTokenCost = 0.0;

    agent.events.on('tokensUsed', (usage) => {
        totalInputTokens += usage.inputTokens;
        totalOutputTokens += usage.outputTokens;
        totalInputTokenCost += usage.inputCost ?? 0.0;
        totalOutputTokenCost += usage.outputCost ?? 0.0;
    });

    agent.events.on('start', () => {
        console.log(bold(blueBright(`▶ [start] agent started with ${agent.model.describeModel()}`)));
    });

    agent.events.on('stop', () => {
        console.log(bold(blueBright(`■ [stop] agent stopped`)));

        console.log(`  Total usage: ` + bold`${totalInputTokens}` + ` input tokens` + ` / ` + bold`${totalOutputTokens}` + ` output tokens`);
        if (totalInputTokenCost > 0 || totalOutputTokenCost > 0) {
            if (agent.model.describeModel().startsWith('claude-code')) {
                console.log(`  Cost: ` + cyanBright`None - using Claude Pro or Max subscription`)
            } else {
                console.log(`  Cost: $${(totalInputTokenCost + totalOutputTokenCost).toFixed(3)}`);
            }
        }
        // Show token usage and cost if available
        // if (totalInputTokenCost > 0 || totalOutputTokenCost > 0) {
        //     console.log(`  Total usage: ` + bold`${totalInputTokens}` + ` input tokens (` + `$${totalInputTokenCost.toFixed(3)}` + `)` + ` / ` + bold`${totalOutputTokens}` + ` output tokens (` + `$${totalOutputTokenCost.toFixed(3)}` + `)`);
        // } else {
        //     console.log(`  Total usage: ` + bold`${totalInputTokens}` + ` input tokens` + ` / ` + bold`${totalOutputTokens}` + ` output tokens`);
        // }
    });

    agent.events.on('thought', (thought: string) => {
        console.log(gray`${thought}`);
        //console.log(gray`⚙︎ ${thought}`);
    });

    agent.events.on('actStarted', (task: string, options: ActOptions) => {
        console.log(bold(cyanBright(`◆ [act] ${task}`)));
    });

    agent.events.on('actionStarted', (action: Action) => {
        const actionDefinition = agent.identifyAction(action);
        console.log(bold`  ${actionDefinition.render(action)}`);
    });
}

export function narrateBrowserAgent(agent: BrowserAgent) {
    narrateAgent(agent);

    agent.browserAgentEvents.on('nav', (url: string) => {
        console.log(bold(cyanBright`⛓︎ [nav] ${url}`));
    });

    agent.browserAgentEvents.on('extractStarted', (instructions: string, schema: z.ZodSchema) => {
        console.log(bold(cyanBright`⛏ [extract] ${instructions}`));
    });
    agent.browserAgentEvents.on('extractDone', (instructions, data) => {
        // console.log has a decent default formatter for arbitrary data e.g. objects
        console.log(data);
        //console.log(blueBright`${JSON.stringify(data, null, 2)}`);
    });
}