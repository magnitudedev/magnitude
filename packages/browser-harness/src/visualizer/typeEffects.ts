import { retryOnError, retryOnErrorIsSuccess } from "../common";
import { BrowserContext, Page } from "playwright";

export class TypeEffectVisual {
    private baseOpacity: number;
    private displayDuration: number;
    private maxKeys: number;

    private page!: Page;

    constructor(baseOpacity: number = 0.6, displayDuration: number = 2000, maxKeys: number = 5) {
        this.baseOpacity = baseOpacity;
        this.displayDuration = displayDuration;
        this.maxKeys = maxKeys;
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

    async setupOnPage() {
        await this.page.evaluate((options: { opacity: number, displayDuration: number, maxKeys: number }) => {
            if ((window as any).__typeEffectsInitialized) {
                return;
            }
            (window as any).__typeEffectsInitialized = true;

            // Create container for key badges
            const container = document.createElement('div');
            container.id = 'type-effects-container';
            container.style.cssText = `
                position: fixed;
                bottom: 40px;
                left: 50%;
                transform: translateX(-50%);
                display: flex;
                gap: 8px;
                align-items: center;
                pointer-events: none;
                z-index: 1000000060;
            `;
            document.body.appendChild(container);

            // Add styles for key badges
            const style = document.createElement('style');
            style.setAttribute('data-type-effects', 'true');
            style.textContent = `
                @keyframes keyIn {
                    0% {
                        opacity: 0;
                        transform: scale(0.5) translateY(20px);
                    }
                    100% {
                        opacity: 1;
                        transform: scale(1) translateY(0);
                    }
                }
                @keyframes keyOut {
                    0% {
                        opacity: 1;
                        transform: scale(1) translateY(0);
                    }
                    100% {
                        opacity: 0;
                        transform: scale(0.8) translateY(10px);
                    }
                }
                .key-badge {
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    min-width: 40px;
                    height: 40px;
                    padding: 0 12px;
                    background: rgba(0, 100, 200, ${options.opacity * 0.9});
                    border: none;
                    border-radius: 6px;
                    font-family: monospace;
                    font-size: 16px;
                    font-weight: bold;
                    color: white;
                    animation: keyIn 0.2s ease-out forwards;
                    white-space: nowrap;
                    user-select: none;
                    -webkit-user-select: none;
                    -moz-user-select: none;
                    -ms-user-select: none;
                }
                .key-badge.fading {
                    animation: keyOut 0.3s ease-out forwards;
                }
            `;
            document.head.appendChild(style);

            // Define special keys to track (excluding modifiers on their own)
            const specialKeys: { [key: string]: string } = {
                'Enter': '↵ Enter',
                'Tab': '⇥ Tab',
                'Delete': '⌦ Del',
                'Escape': '⎋ Esc',
                'ArrowUp': '↑',
                'ArrowDown': '↓',
                'ArrowLeft': '←',
                'ArrowRight': '→',
                'Home': '⇱ Home',
                'End': '⇲ End',
                'PageUp': '⇞ PgUp',
                'PageDown': '⇟ PgDn',
                'Insert': 'Ins',
            };

            // Active badges
            const activeBadges: { element: HTMLElement, timeout: ReturnType<typeof setTimeout> }[] = [];

            // Handle keydown events
            document.addEventListener('keydown', (e) => {
                const key = e.key;
                
                // Check if we have meaningful modifiers (Ctrl or Cmd)
                const hasMeaningfulModifiers = e.ctrlKey || e.metaKey;
                
                // For modifier combos, we want to show them with ANY key (including letters/numbers)
                if (hasMeaningfulModifiers) {
                    // Skip if it's just a modifier key by itself
                    if (['Control', 'Alt', 'Shift', 'Meta'].includes(key)) {
                        return;
                    }
                    
                    // Build the combo display
                    let modifiers = [];
                    if (e.ctrlKey) modifiers.push('Ctrl');
                    if (e.metaKey) modifiers.push('⌘');
                    // Include Shift/Alt only if Ctrl or Cmd is also pressed
                    if ((e.ctrlKey || e.metaKey) && e.shiftKey) modifiers.push('Shift');
                    if ((e.ctrlKey || e.metaKey) && e.altKey) modifiers.push('Alt');
                    
                    // Get the key display
                    let keyDisplay = specialKeys[key] || key.toUpperCase();
                    if (key.length === 1) {
                        keyDisplay = key.toUpperCase();
                    }
                    
                    const displayText = modifiers.join('+') + '+' + keyDisplay;
                    
                    // Create badge
                    createBadge(displayText);
                } else {
                    // No modifiers - only show if it's a special key
                    const keyDisplay = specialKeys[key];
                    if (keyDisplay) {
                        createBadge(keyDisplay);
                    }
                }
            });

            function createBadge(displayText: string) {
                // Create badge element
                const badge = document.createElement('div');
                badge.className = 'key-badge';
                badge.textContent = displayText;
                
                // Add to container
                container.appendChild(badge);

                // Set up removal timeout
                const timeout = setTimeout(() => {
                    badge.classList.add('fading');
                    setTimeout(() => {
                        badge.remove();
                        // Remove from active badges
                        const index = activeBadges.findIndex(b => b.element === badge);
                        if (index !== -1) {
                            activeBadges.splice(index, 1);
                        }
                    }, 300);
                }, options.displayDuration);

                // Track active badge
                activeBadges.push({ element: badge, timeout });

                // Remove oldest badges if we exceed max
                while (activeBadges.length > options.maxKeys) {
                    const oldest = activeBadges.shift();
                    if (oldest) {
                        clearTimeout(oldest.timeout);
                        oldest.element.classList.add('fading');
                        setTimeout(() => oldest.element.remove(), 300);
                    }
                }
            }

            // Clean up on page hide
            document.addEventListener('visibilitychange', () => {
                if (document.hidden) {
                    activeBadges.forEach(({ element, timeout }) => {
                        clearTimeout(timeout);
                        element.remove();
                    });
                    activeBadges.length = 0;
                }
            });
        }, { opacity: this.baseOpacity, displayDuration: this.displayDuration, maxKeys: this.maxKeys });
    }
}