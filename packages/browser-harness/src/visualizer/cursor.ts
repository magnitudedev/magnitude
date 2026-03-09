import { retryOnError, retryOnErrorIsSuccess } from "../common";
import logger from "../logger";
import { Page } from "playwright";

export class CursorVisual {
    /**
     * Manages the visual indicator for actions on a page
     */
    private page!: Page;
    private visualElementId: string = 'action-visual-indicator';
    private lastPosition: { x: number; y: number } | null = null;

    constructor() {
        //this.page = page;
    }

    async setActivePage(page: Page) {
        this.page = page;

        page.on('load', async () => {
            await retryOnErrorIsSuccess(
                this.setupOnPage.bind(this),
                { mode: 'retry_all', delayMs: 200, retryLimit: 10 }
            );
        });

        await retryOnErrorIsSuccess(
            this.setupOnPage.bind(this),
            { mode: 'retry_all', delayMs: 200, retryLimit: 5 }
        );
    }

    async move(x: number, y: number): Promise<void> {
        // Store the position
        this.lastPosition = { x, y };
        // Create or update the mouse pointer visual, showing the click effect
        await this._drawVisual(x, y, false);
        // The pointer visual takes 0.3s on the transition, but awaiting script evaluation does not wait for this to complete.
        // So we wait 300ms manually.
        await this.page.waitForTimeout(300);
    }

    async setupOnPage(): Promise<void> {
        if (this.lastPosition) {
            // Redraw the visual without the click effect
            await this._drawVisual(this.lastPosition.x, this.lastPosition.y, false);
        }
    }

    // Internal method to handle the actual drawing logic
    private async _drawVisual(x: number, y: number, showClickEffect: boolean): Promise<void> {
        try {
            await this.page.evaluate(
                ({ x, y, id, showClickEffect }) => {
                    // Use viewport coordinates directly (no scroll adjustment for fixed positioning)
                    const viewportX = x;
                    const viewportY = y;

                    // Document coordinates for the click effect circle (which uses absolute positioning)
                    const docX = x + window.scrollX;
                    const docY = y + window.scrollY;

                    // --- Create Expanding/Fading Circle (Optional) ---
                    if (showClickEffect) {
                        const circle = document.createElement('div');
                        circle.style.position = 'absolute';
                        circle.style.left = `${docX}px`;
                        circle.style.top = `${docY}px`;
                        circle.style.borderRadius = '50%';
                        circle.style.backgroundColor = '#026aa1'; // Blue color
                        circle.style.width = '0px';
                        circle.style.height = '0px';
                        circle.style.transform = 'translate(-50%, -50%)'; // Center on (x, y)
                        circle.style.pointerEvents = 'none';
                        circle.style.zIndex = '9998'; // Below the pointer
                        circle.style.opacity = '0.7'; // Initial opacity
                        document.body.appendChild(circle);

                        // Animate the circle
                        const animation = circle.animate([
                            { width: '0px', height: '0px', opacity: 0.7 }, // Start state
                            { width: '50px', height: '50px', opacity: 0 }  // End state
                        ], {
                            duration: 500, // 500ms duration
                            easing: 'ease-out'
                        });

                        // Remove circle after animation
                        animation.onfinish = () => {
                            circle.remove();
                        };
                    }

                    // --- Pointer Logic (Always runs) ---
                    // Check if the visual indicator already exists
                    let pointerElement = document.getElementById(id);
                    
                    // If it doesn't exist, create it with all necessary styling
                    if (!pointerElement) {
                        pointerElement = document.createElement('div');
                        pointerElement.id = id;
                        pointerElement.style.position = 'fixed';  // Use fixed positioning for viewport-relative
                        pointerElement.style.width = '32px';
                        pointerElement.style.height = '32px';
                        pointerElement.style.zIndex = '2147483647';  // Max z-index
                        pointerElement.style.pointerEvents = 'none'; // Don't interfere with actual clicks
                        // Notice that transition is 300ms
                        pointerElement.style.transition = 'left 0.3s cubic-bezier(0.25, 0.1, 0.25, 1), top 0.3s cubic-bezier(0.25, 0.1, 0.25, 1)';

                        // Create SVG using DOM methods to avoid Trusted Types issues
                        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
                        svg.setAttribute('width', '32');
                        svg.setAttribute('height', '32');
                        svg.setAttribute('viewBox', '0 0 113.50408 99.837555');

                        const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                        g.setAttribute('transform', 'translate(-413.10686,-501.19661)');

                        const path1 = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                        path1.setAttribute('style', 'fill:#026aa1;fill-opacity:1;stroke:#000000;stroke-width:0');
                        path1.setAttribute('d', 'm 416.1069,504.1966 52.47697,93.83813 8.33253,-57.61019 z');

                        const path2 = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                        path2.setAttribute('style', 'fill:#0384c7;fill-opacity:1;stroke:#000000;stroke-width:0');
                        path2.setAttribute('d', 'm 416.1069,504.1966 60.8095,36.22794 46.69517,-34.75524 z');

                        const path3 = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                        path3.setAttribute('style', 'fill:#0384c7;fill-opacity:0;stroke:#000000;stroke-width:6;stroke-linecap:round;stroke-linejoin:round');
                        path3.setAttribute('d', 'm 416.1069,504.19658 52.47698,93.83813 8.33252,-57.61019 46.69517,-34.75521 -107.50467,-1.47273');

                        g.appendChild(path1);
                        g.appendChild(path2);
                        g.appendChild(path3);
                        svg.appendChild(g);
                        pointerElement.appendChild(svg);

                        document.body.appendChild(pointerElement);
                    }
                    
                    //pointerElement.style.display = 'none';

                    // Update position - use viewport coordinates for fixed positioning
                    // Set the top-left corner to (viewportX, viewportY) and then translate by (-1px, -3px)
                    // to align the pointer tip (approx. at 1.27, 4.17 within the SVG) with the click point.
                    pointerElement.style.left = `${viewportX}px`;
                    pointerElement.style.top = `${viewportY}px`;
                    pointerElement.style.transform = 'translate(-1px, -3px)';
                },
                { x, y, id: this.visualElementId, showClickEffect }
            );
        } catch (error: unknown) {
            // For example when:
            // TypeError: Failed to set the 'innerHTML' property on 'Element': This document requires 'TrustedHTML' assignment.
            logger.trace(`Failed to draw visual: ${(error as Error).message}`);
        }
    }

    async hide(): Promise<void> {
        try {
            await this.page.evaluate((id) => {
                const element = document.getElementById(id);
                if (element) {
                    element.style.display = 'none';
                }
            }, this.visualElementId);
        } catch {
            logger.trace(`Failed to hide pointer`);
        }
    }

    async show(): Promise<void> {
        try {
            await this.page.evaluate((id) => {
                const element = document.getElementById(id);
                if (element) {
                    // Revert to the default display value (usually 'block' for a div)
                    element.style.display = ''; 
                }
            }, this.visualElementId);
        } catch {
            logger.trace(`Failed to show pointer`);
        }
    }

    // async removeActionVisuals(): Promise<void> {
    //     // Remove the visual indicator
    //     await this.page.evaluate((id) => {
    //         const element = document.getElementById(id);
    //         if (element) {
    //             element.remove();
    //         }
    //     }, this.visualElementId);
    // }
}
