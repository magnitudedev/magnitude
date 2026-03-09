// todo: use memory Image type
export type Base64Image = `data:image/${'jpeg'|'png'|'gif'};base64,${string}`;

// export interface Screenshot {
//     // b64-encoded PNG image
//     image: Base64Image,
//     dimensions: {
//         width: number,
//         height: number
//     }
// }

// export interface WebAction {
//     variant: ActionVariant
// }

export type WebAction = NavigateWebAction | ClickWebAction | TypeWebAction | ScrollWebAction | SwitchTabWebAction;

// Currently only emitted synthetically
export interface NavigateWebAction {
    variant: 'load'
    url: string
}

export interface ClickWebAction {
    variant: 'click'
    x: number
    y: number
}

export interface TypeWebAction {
    variant: 'type'
    x: number
    y: number
    content: string
}

export interface ScrollWebAction {
    variant: 'scroll',
    x: number,
    y: number,
    deltaX: number,
    deltaY: number
}

export type SwitchTabWebAction = {
    variant: 'tab',
    index: number
}

// export type ActionVariant = 'click' | 'type';


export interface PixelCoordinate {
    x: number,
    y: number
}

// for back-compat, not all implemented yet

