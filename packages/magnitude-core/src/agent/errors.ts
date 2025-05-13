import { FailureDescriptor, generateSimpleFailureString } from "../common";


export class AgentError extends Error {
    public readonly failure: FailureDescriptor;

    constructor(failure: FailureDescriptor) {
        super(generateSimpleFailureString(failure))
        this.name = "AgentError";
        this.failure = failure;
    }
}

// // Agent will only throw these types of errors

// import { ActionIngredient } from "./recipe/types";
// import { WebAction } from "./web/types";

// export class TestCaseError extends Error {};

// // Base class for wrapping errors
// class WrapperError extends TestCaseError {
//     public readonly originalError: Error;

//     constructor(message: string, originalError: Error) {
//         super(`${message} - ${originalError.message}`);
//         this.name = this.constructor.name;
//         this.originalError = originalError;

//         // Preserve the stack trace
//         if (Error.captureStackTrace) {
//             Error.captureStackTrace(this, this.constructor);
//         }
//     }
// }

// export class NavigationError extends WrapperError {
//     public readonly url: string;

//     constructor(url: string, originalError: Error) {
//         super(`Failed to navigate to URL: ${url}`, originalError);
//         this.url = url;
//     }
// }

// export class ActionExecutionError extends WrapperError {
//     public readonly action: WebAction;
//     public readonly statusCode?: number;

//     constructor(action: WebAction, originalError: Error) {
//         super(`Browser encountered error while executing action: ${action}`, originalError);
//         this.action = action;
//     }
// }

// export class ActionConversionError extends WrapperError {
//     public readonly action: ActionIngredient;
//     public readonly statusCode?: number;

//     constructor(action: ActionIngredient, originalError: Error) {
//         super(`Micro failed to convert action (does the target line up with the screenshot?): ${action}`, originalError);
//         this.action = action;
//     }
// }
