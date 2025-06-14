// import { StepOptions } from "@/types";
// import { Agent } from ".";
// import { ActionIntent } from "@/actions/types";

// export class Magnus {
//     /**
//      * Wrapper to expose only relevant methods of TestCaseAgent to test case writers
//      */
//     private agent: Agent;

//     constructor(agent: Agent) {
//         this.agent = agent;
//     }

//     // High level AI operations

//     async step(description: string, options: StepOptions = {}): Promise<void> {
//         return await this.agent.step(description, options);
//     }

//     async check(description: string): Promise<void> {
//         return await this.agent.check(description);
//     }

//     // Low level AI operations

//     async click(target: string): Promise<void> {
//         await this.agent.exec({
//             variant: 'click',
//             target: target
//         });
//     }

//     async type(target: string, content: string): Promise<void> {
//         await this.agent.exec({
//             variant: 'type',
//             target: target,
//             content: content
//         });
//     }

//     async exec(action: ActionIntent) {
//         await this.agent.exec(action);
//     }

//     // tmp
//     // get harness() {
//     //     return this.agent.harness;
//     // }
// }