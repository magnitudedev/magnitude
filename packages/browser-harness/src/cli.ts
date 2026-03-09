#!/usr/bin/env node

import { WebHarness, BrowserProvider } from './index';
import { Command } from '@commander-js/extra-typings';
import logger from './logger';

const program = new Command();

program
  .name('magnitude-harness')
  .description('CLI for testing magnitude-harness browser automation')
  .version('0.1.0');

program
  .command('demo')
  .description('Run a simple demo showing harness capabilities')
  .option('--headless', 'Run in headless mode', false)
  .option('--url <url>', 'URL to navigate to', 'https://example.com')
  .action(async (options) => {
    logger.info('Starting harness demo...');

    const browserProvider = BrowserProvider.getInstance();
    const context = await browserProvider.newContext({
      launchOptions: {
        headless: options.headless
      }
    });

    const harness = new WebHarness(context, {
      virtualScreenDimensions: { width: 1024, height: 768 },
      switchTabsOnActivity: true,
      visuals: {
        showCursor: true,
        showClickRipple: true,
        showTypeEffects: true
      }
    });

    try {
      await harness.start();
      logger.info(`Navigating to ${options.url}...`);
      await harness.navigate(options.url);

      logger.info('Taking screenshot...');
      const screenshot = await harness.screenshot();
      const dims = await screenshot.getDimensions();
      logger.info(`Screenshot captured: ${dims.width}x${dims.height}`);

      logger.info('Getting tab state...');
      const tabState = await harness.retrieveTabState();
      logger.info(`Active tab: ${tabState.activeTab}, Total tabs: ${tabState.tabs.length}`);

      if (!options.headless) {
        logger.info('Browser will stay open. Press Ctrl+C to exit.');
        // Keep process alive
        await new Promise(() => {});
      } else {
        logger.info('Demo complete!');
        await harness.stop();
        await context.close();
      }
    } catch (error) {
      logger.error('Error during demo:', error);
      await harness.stop();
      await context.close();
      process.exit(1);
    }
  });

program
  .command('navigate <url>')
  .description('Navigate to a URL and take a screenshot')
  .option('--headless', 'Run in headless mode', false)
  .option('--output <path>', 'Save screenshot to file')
  .action(async (url, options) => {
    const browserProvider = BrowserProvider.getInstance();
    const context = await browserProvider.newContext({
      launchOptions: {
        headless: options.headless
      }
    });

    const harness = new WebHarness(context);

    try {
      await harness.start();
      logger.info(`Navigating to ${url}...`);
      await harness.navigate(url);

      const screenshot = await harness.screenshot();
      const dims = await screenshot.getDimensions();
      logger.info(`Screenshot: ${dims.width}x${dims.height}`);

      if (options.output) {
        await screenshot.saveToFile(options.output);
        logger.info(`Screenshot saved to ${options.output}`);
      }

      await harness.stop();
      await context.close();
    } catch (error) {
      logger.error('Error:', error);
      process.exit(1);
    }
  });

program
  .command('interactive')
  .description('Start an interactive browser session')
  .option('--url <url>', 'Initial URL', 'https://google.com')
  .action(async (options) => {
    const browserProvider = BrowserProvider.getInstance();
    const context = await browserProvider.newContext({
      launchOptions: {
        headless: false
      }
    });

    const harness = new WebHarness(context, {
      visuals: {
        showCursor: true,
        showClickRipple: true,
        showTypeEffects: true
      }
    });

    try {
      await harness.start();
      if (options.url) {
        await harness.navigate(options.url);
      }

      logger.info('Interactive session started. Browser will stay open.');
      logger.info('Available commands:');
      logger.info('  - Browser is ready for interaction');
      logger.info('  - Press Ctrl+C to exit');

      // Set up graceful shutdown
      process.on('SIGINT', async () => {
        logger.info('\nShutting down...');
        await harness.stop();
        await context.close();
        process.exit(0);
      });

      // Keep process alive
      await new Promise(() => {});
    } catch (error) {
      logger.error('Error:', error);
      process.exit(1);
    }
  });

program.parse();
