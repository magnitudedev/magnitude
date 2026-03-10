import type { Page } from 'playwright';
import getShadowDOMInputAdapterScript from './scripts/shadowDOMInputAdapter';
import logger from './logger';

export class DOMTransformer {
    private initializedPages = new WeakSet<Page>(); // Track pages for which 'load' listener is set

    constructor() {}

    public setActivePage(newPage: Page) {
        // Only add the 'load' listener if we haven't done so for this specific Page object instance.
        if (!this.initializedPages.has(newPage)) {
            newPage.on('load', async () => {
                // Pass 'newPage' (the page that triggered the 'load' event) to setupScriptForPage.
                await this.setupScriptForPage(newPage);
            });
            this.initializedPages.add(newPage); // Mark this Page object as having its 'load' listener set up.
        }
    }

    public async setupScriptForPage(targetPage: Page) {
        try {
            // Check if a marker for the script already exists on the page for this load cycle.
            const scriptAlreadyInjected = await targetPage.evaluate(() => {
                return (window as any).__magnitudeShadowDOMAdapterInjected === true;
            }).catch(() => false); // If evaluate fails (e.g., page closed), assume not injected.

            if (scriptAlreadyInjected) {
                logger.debug('Select manager script already present on this page load.');
                return;
            }

            // Get the script as a string from the separate file
            const scriptFnString = getShadowDOMInputAdapterScript();

            // Evaluate the script function in the browser and set the marker.
            // We need to wrap it in a self-executing function
            await targetPage.evaluate(`
                (${scriptFnString})();
                window.__magnitudeShadowDOMAdapterInjected = true;
            `);

            logger.debug(`Script injected into page: ${targetPage.url()}`);
        } catch (error) {
            const url = targetPage.isClosed() ? '[closed page]' : targetPage.url();
            logger.warn(`Error injecting script into ${url}: ${(error as Error).message}`);
        }
    }
}
