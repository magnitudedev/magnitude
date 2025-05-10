import { ClickWebAction, NavigateWebAction, ScrollWebAction, SwitchTabWebAction, TypeWebAction } from "@/web/types";
import { ClickIntent, TypeIntent, ScrollIntent, SwitchTabIntent } from "@/intents/types";

//'load' | 'click' | 'hover' | 'type' | 'scroll' | 'wait' | 'back';

export type ActionDescriptor = NavigateActionDescriptor | ClickActionDescriptor | TypeActionDescriptor | ScrollActionDescriptor | SwitchTabDescriptor;
export type ActionVariant = ActionDescriptor['variant'];

export type NavigateActionDescriptor = NavigateWebAction & { screenshot: string };
export type ClickActionDescriptor = ClickIntent & ClickWebAction & { screenshot: string };
export type TypeActionDescriptor = TypeIntent & TypeWebAction & { screenshot: string };
export type ScrollActionDescriptor = ScrollIntent & ScrollWebAction & { screenshot: string };
export type SwitchTabDescriptor = SwitchTabIntent & SwitchTabWebAction & { screenshot: string };

//const a: ClickActionDescriptor = { variant: 'click', x: 1, y: 2, target: ''}
