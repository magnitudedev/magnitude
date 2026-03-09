import { BrowserContext, Page } from "playwright";
import { CursorVisual } from "./cursor";
import { MouseEffectVisual } from "./mouseEffects";
import { TypeEffectVisual } from "./typeEffects";

export interface ActionVisualizerOptions {
    showCursor?: boolean,
    showHoverCircle?: boolean; // mouse effect option
    showClickRipple?: boolean; // mouse effect option
    showDragLine?: boolean; // mouse effect option
    showTypeEffects?: boolean,
}

export class ActionVisualizer {
    private options: Required<ActionVisualizerOptions>;
    private context: BrowserContext;
    private page!: Page;

    private cursor: CursorVisual;
    private mouseEffects: MouseEffectVisual;
    private typeEffects: TypeEffectVisual;
    

    constructor(context: BrowserContext, options: ActionVisualizerOptions) {
        this.context = context;
        this.options = {
            showCursor: options.showCursor ?? true,
            showHoverCircle: options.showHoverCircle ?? false,
            showClickRipple: options.showClickRipple ?? true,
            showDragLine: options.showDragLine ?? false,
            showTypeEffects: options.showTypeEffects ?? true
        };

        this.cursor = new CursorVisual();
        this.mouseEffects = new MouseEffectVisual(this.options);
        this.typeEffects = new TypeEffectVisual();
    }

    async setup() {
        // await this.mouseEffects.setContext(this.context);
        // if (this.options.showTypeEffects) {
        //     await this.typeEffects.setContext(this.context);
        // }
        //this.showAll();
    }

    async setActivePage(page: Page) {
        this.page = page;
        await this.cursor.setActivePage(page);
        await this.mouseEffects.setActivePage(page);
        await this.typeEffects.setActivePage(page);
    }

    async moveVirtualCursor(x: number, y: number) {
        // Takes like 300ms for smooth anim
        await this.cursor.move(x, y);
    }

    async hideAll() {
        await this.cursor.hide();
    }

    async showAll() {
        await this.cursor.show();
    }
}