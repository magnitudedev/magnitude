import { retryOnError, retryOnErrorIsSuccess } from "../common";
import { BrowserContext, Page } from "playwright";

interface MouseEffectOptions {
    baseOpacity?: number;
    showHoverCircle?: boolean;
    showClickRipple?: boolean;
    showDragLine?: boolean;
}

export class MouseEffectVisual {
    private baseOpacity: number;
    private showHoverCircle: boolean;
    private showClickRipple: boolean;
    private showDragLine: boolean;

    private page!: Page;

    constructor(options: MouseEffectOptions = {}) {
        this.baseOpacity = options.baseOpacity ?? 0.5;
        this.showHoverCircle = options.showHoverCircle ?? true;
        this.showClickRipple = options.showClickRipple ?? true;
        this.showDragLine = options.showDragLine ?? true;
    }

    async setActivePage(page: Page) {
        this.page = page;

        page.on('load', async () => {
            // Clear the initialized flag on navigation to ensure re-initialization
            await page.evaluate(() => {
                (window as any).__mouseEffectsInitialized = false;
            }).catch(() => {}); // Ignore errors
            
            await retryOnErrorIsSuccess(
                this.setupOnPage.bind(this),
                { mode: 'retry_all', delayMs: 200, retryLimit: 10 }
            );
        });

        await retryOnErrorIsSuccess(
            this.setupOnPage.bind(this),
            { mode: 'retry_all', delayMs: 200, retryLimit: 10 }
        );
    }

    async setupOnPage() {
        await this.page.evaluate((options: { opacity: number, showHoverCircle: boolean, showClickRipple: boolean, showDragLine: boolean }) => {
            if ((window as any).__mouseEffectsInitialized) {
                return;
            }
            (window as any).__mouseEffectsInitialized = true;

            // Create custom cursor element only if enabled
            let cursor: HTMLElement | null = null;
            if (options.showHoverCircle) {
                cursor = document.createElement('div');
                cursor.id = 'custom-cursor';
                cursor.style.cssText = `
                    position: fixed;
                    width: 20px;
                    height: 20px;
                    border: 2px solid rgba(0, 150, 255, ${options.opacity});
                    border-radius: 50%;
                    background: rgba(0, 150, 255, ${options.opacity * 0.5});
                    pointer-events: none;
                    z-index: 1000000050;
                    transition: transform 0.1s ease-out;
                `;
                document.body.appendChild(cursor);
            }

            // Create drag line element only if enabled
            let dragLine: SVGSVGElement | null = null;
            let line: SVGLineElement | null = null;
            if (options.showDragLine) {
                dragLine = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
                dragLine.id = 'drag-line';
                dragLine.style.cssText = `
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100vw;
                    height: 100vh;
                    pointer-events: none;
                    z-index: 1000000030;
                    overflow: visible;
                `;
                line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                line.setAttribute('stroke', '#0096ff');
                line.setAttribute('stroke-width', '2');
                line.setAttribute('stroke-dasharray', '5,5');
                line.setAttribute('opacity', '0.8');
                line.style.display = 'none';
                dragLine.appendChild(line);
                document.body.appendChild(dragLine);
            }

            let isMouseDown = false;
            let dragStartX = 0;
            let dragStartY = 0;
            const activeRipples = new Set<HTMLElement>();

            // Track mouse movement
            document.addEventListener('mousemove', (e) => {
                // Update cursor position - always center on mouse
                if (cursor) {
                    cursor.style.left = e.clientX - 10 + 'px';
                    cursor.style.top = e.clientY - 10 + 'px';
                }

                // Update drag line if dragging
                if (isMouseDown && line) {
                    line.setAttribute('x1', dragStartX.toString());
                    line.setAttribute('y1', dragStartY.toString());
                    line.setAttribute('x2', e.clientX.toString());
                    line.setAttribute('y2', e.clientY.toString());
                    line.style.display = 'block';
                }
            });

            // Mouse down - start drag and shrink cursor
            document.addEventListener('mousedown', (e) => {
                isMouseDown = true;
                dragStartX = e.clientX;
                dragStartY = e.clientY;

                // Use transform scale instead of changing width/height
                if (cursor) {
                    cursor.style.transform = 'scale(0.5)';
                    cursor.style.background = `rgba(0, 150, 255, ${options.opacity * 1.5})`;
                    cursor.style.border = `2px solid rgba(0, 150, 255, ${options.opacity * 2})`;
                }
            });

            // Mouse up - end drag and restore cursor
            document.addEventListener('mouseup', (e) => {
                isMouseDown = false;

                // Hide drag line
                if (line) {
                    line.style.display = 'none';
                }

                // Restore cursor
                if (cursor) {
                    cursor.style.transform = 'scale(1)';
                    cursor.style.background = `rgba(0, 150, 255, ${options.opacity * 0.5})`;
                    cursor.style.border = `2px solid rgba(0, 150, 255, ${options.opacity})`;
                }
            });

            // Visualize clicks
            document.addEventListener('click', (e) => {
                if (options.showClickRipple) {
                    // Add ripple animation styles once
                    if (!document.querySelector('style[data-cursor-effects]')) {
                        const style = document.createElement('style');
                        style.setAttribute('data-cursor-effects', 'true');
                        style.textContent = `
                            @keyframes ripple {
                                0% {
                                    transform: scale(0);
                                    opacity: 1;
                                }
                                100% {
                                    transform: scale(2);
                                    opacity: 0;
                                }
                            }
                            .click-ripple {
                                position: fixed;
                                width: 60px;
                                height: 60px;
                                border: 3px solid #0096ff;
                                border-radius: 50%;
                                pointer-events: none;
                                z-index: 1000000040;
                                transform: scale(0);
                                opacity: 0;
                            }
                            .click-ripple.active {
                                animation: ripple 0.6s ease-out forwards;
                            }
                        `;
                        document.head.appendChild(style);
                    }

                    // Create ripple effect
                    const ripple = document.createElement('div');
                    ripple.className = 'click-ripple';
                    ripple.style.left = `${e.clientX - 30}px`;
                    ripple.style.top = `${e.clientY - 30}px`;
                    
                    document.body.appendChild(ripple);
                    activeRipples.add(ripple);
                    
                    // Trigger animation on next frame to ensure proper initialization
                    requestAnimationFrame(() => {
                        ripple.classList.add('active');
                    });

                    // Remove ripple after animation
                    setTimeout(() => {
                        ripple.remove();
                        activeRipples.delete(ripple);
                    }, 600);
                }

                // Flash the cursor only if not already scaled down
                if (cursor) {
                    const currentTransform = cursor.style.transform;
                    if (currentTransform !== 'scale(0.5)') {
                        cursor.style.background = `rgba(0, 150, 255, ${options.opacity * 1.5})`;
                        cursor.style.border = `2px solid rgba(0, 150, 255, 1)`;
                        cursor.style.transform = 'scale(1.5)';

                        setTimeout(() => {
                            cursor.style.background = `rgba(0, 150, 255, ${options.opacity * 0.5})`;
                            cursor.style.border = `2px solid rgba(0, 150, 255, ${options.opacity})`;
                            cursor.style.transform = 'scale(1)';
                        }, 200);
                    }
                }
            });

            // Also track right clicks
            document.addEventListener('contextmenu', (e) => {
                // Don't prevent default - let context menu show

                if (options.showClickRipple) {
                    // Create ripple for right click (reuse same styles)
                    const ripple = document.createElement('div');
                    ripple.className = 'click-ripple';
                    ripple.style.left = `${e.clientX - 30}px`;
                    ripple.style.top = `${e.clientY - 30}px`;
                    
                    document.body.appendChild(ripple);
                    activeRipples.add(ripple);
                    
                    // Trigger animation on next frame
                    requestAnimationFrame(() => {
                        ripple.classList.add('active');
                    });
                    
                    setTimeout(() => {
                        ripple.remove();
                        activeRipples.delete(ripple);
                    }, 600);
                }
            });

            // Scroll visualization
            let scrollTimeout: ReturnType<typeof setTimeout> | undefined;
            let scrollArrow: HTMLElement | null = null;

            // Create the single scroll arrow
            scrollArrow = document.createElement('div');
            scrollArrow.style.cssText = `
                position: fixed;
                width: 40px;
                height: 40px;
                font-size: 24px;
                color: #0096ff;
                text-align: center;
                line-height: 40px;
                pointer-events: none;
                z-index: 1000000045;
                display: none;
                transition: transform 0.1s ease-out, opacity 0.2s ease-out;
            `;
            document.body.appendChild(scrollArrow);

            // Handle both wheel and scroll events
            const handleScroll = (e: WheelEvent) => {
                if (!scrollArrow) return;
                
                // Skip if no vertical scroll
                if (Math.abs(e.deltaY) < 0.5) return;
                
                // Clear existing timeout
                if (scrollTimeout) {
                    clearTimeout(scrollTimeout);
                }

                // Update arrow
                const scrollDirection = e.deltaY > 0 ? 'down' : 'up';
                scrollArrow.textContent = scrollDirection === 'down' ? '▼' : '▲';
                
                // Position above/below cursor based on direction
                // Arrow is 40px tall, cursor is ~20px, add 10px spacing
                const offset = scrollDirection === 'down' ? 30 : -30;
                scrollArrow.style.left = `${e.clientX - 20}px`;
                scrollArrow.style.top = `${e.clientY + offset - 20}px`; // -20 to account for arrow center
                
                // Scale based on deltaY magnitude
                const scale = 0.8 + Math.min(Math.abs(e.deltaY) / 10, 0.4);
                scrollArrow.style.transform = `scale(${scale})`;
                scrollArrow.style.opacity = '0.4';
                scrollArrow.style.display = 'block';

                // Hide after scroll stops
                scrollTimeout = setTimeout(() => {
                    if (scrollArrow) {
                        scrollArrow.style.opacity = '0';
                        setTimeout(() => {
                            if (scrollArrow) scrollArrow.style.display = 'none';
                        }, 200);
                    }
                }, 150);
            };

            // Add event listeners for both wheel and scroll events
            document.addEventListener('wheel', handleScroll, { passive: true });
            
            // Also listen to scroll events on window and document.body for better compatibility
            window.addEventListener('wheel', handleScroll, { passive: true });
            document.body.addEventListener('wheel', handleScroll, { passive: true });

            // Hide cursor when it leaves the viewport
            document.addEventListener('mouseleave', () => {
                if (cursor) {
                    cursor.style.display = 'none';
                }
                // Clean up any active ripples
                activeRipples.forEach(ripple => ripple.remove());
                activeRipples.clear();
                // Hide scroll arrow
                if (scrollArrow) {
                    scrollArrow.style.display = 'none';
                }
            });

            document.addEventListener('mouseenter', () => {
                if (cursor) {
                    cursor.style.display = 'block';
                }
            });
        }, { opacity: this.baseOpacity, showHoverCircle: this.showHoverCircle, showClickRipple: this.showClickRipple, showDragLine: this.showDragLine });
    }
}