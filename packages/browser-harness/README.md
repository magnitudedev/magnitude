# magnitude-harness

Standalone browser automation library built on Playwright. No AI dependencies - just browser control.

## Installation

```bash
npm install magnitude-harness
```

## Quick Start

```typescript
import { WebHarness, BrowserProvider } from 'magnitude-harness';

const context = await BrowserProvider.getInstance().newContext();
const harness = new WebHarness(context);

await harness.start();
await harness.navigate('https://example.com');
await harness.click({ x: 100, y: 200 });
await harness.type({ content: 'Hello world' });

const screenshot = await harness.screenshot();
await screenshot.saveToFile('screenshot.png');

await harness.stop();
await context.close();
```

## Examples

### Form Automation

```typescript
import { WebHarness, BrowserProvider } from 'magnitude-harness';

async function fillForm() {
  const context = await BrowserProvider.getInstance().newContext({
    launchOptions: { headless: false }
  });

  const harness = new WebHarness(context, {
    visuals: {
      showCursor: true,
      showClickRipple: true
    }
  });

  await harness.start();
  await harness.navigate('https://example.com/login');

  // Click email field and type
  await harness.clickAndType({ x: 300, y: 200, content: 'user@example.com' });

  // Tab to password field and type
  await harness.type({ content: '<tab>mypassword<enter>' });

  // Wait for navigation
  await harness.waitForStability();

  await harness.stop();
}
```

### Screenshot Capture

```typescript
import { WebHarness, BrowserProvider } from 'magnitude-harness';

async function capturePages(urls: string[]) {
  const context = await BrowserProvider.getInstance().newContext({
    launchOptions: { headless: true }
  });

  const harness = new WebHarness(context);
  await harness.start();

  for (const url of urls) {
    await harness.navigate(url);

    const screenshot = await harness.screenshot();
    const dims = await screenshot.getDimensions();
    console.log(`Captured ${url}: ${dims.width}x${dims.height}`);

    const filename = url.replace(/[^a-z0-9]/gi, '_') + '.png';
    await screenshot.saveToFile(filename);
  }

  await harness.stop();
  await context.close();
}
```

### Multi-Tab Handling

```typescript
import { WebHarness, BrowserProvider } from 'magnitude-harness';

async function multiTab() {
  const context = await BrowserProvider.getInstance().newContext();
  const harness = new WebHarness(context, {
    switchTabsOnActivity: true  // Auto-switch when new tabs open
  });

  await harness.start();
  await harness.navigate('https://example.com');

  // Click a link that opens in new tab
  await harness.click({ x: 400, y: 300 });

  // Check tab state
  const tabState = await harness.retrieveTabState();
  console.log(`Active tab: ${tabState.activeTab}`);
  console.log(`Total tabs: ${tabState.tabs.length}`);

  // Switch back to first tab
  await harness.switchTab({ index: 0 });

  await harness.stop();
}
```

### Scrolling and Dragging

```typescript
await harness.navigate('https://example.com/long-page');

// Scroll down 500px at center of page
await harness.scroll({ x: 512, y: 384, deltaX: 0, deltaY: 500 });

// Drag from one point to another
await harness.drag({ x1: 100, y1: 100, x2: 300, y2: 300 });
```

### Visual Feedback

Show cursor, clicks, and typing indicators for demos or debugging:

```typescript
const harness = new WebHarness(context, {
  visuals: {
    showCursor: true,       // Animated cursor that follows actions
    showClickRipple: true,  // Ripple effect on clicks
    showHoverCircle: false, // Circle on hover
    showDragLine: false,    // Line showing drag path
    showTypeEffects: true   // Typing indicator
  }
});
```

All visual options default to sensible values (`showCursor`, `showClickRipple`, and `showTypeEffects` are on by default).

### Virtual Screen Dimensions

For AI agents that expect specific screen sizes:

```typescript
const harness = new WebHarness(context, {
  virtualScreenDimensions: { width: 1280, height: 720 }
});

// Coordinates will be transformed from virtual to actual viewport
await harness.click({ x: 640, y: 360 });  // Center of virtual screen
```

## API

### WebHarness

| Method | Description |
|--------|-------------|
| `start()` | Initialize the harness |
| `stop()` | Clean up resources |
| `navigate(url)` | Navigate to URL |
| `click({ x, y })` | Click at coordinates |
| `rightClick({ x, y })` | Right-click at coordinates |
| `doubleClick({ x, y })` | Double-click at coordinates |
| `type({ content })` | Type text (supports `<enter>`, `<tab>`) |
| `clickAndType({ x, y, content })` | Click then type |
| `scroll({ x, y, deltaX, deltaY })` | Scroll at position |
| `drag({ x1, y1, x2, y2 })` | Drag between points |
| `screenshot()` | Capture screenshot as `Image` |
| `switchTab({ index })` | Switch to tab |
| `newTab()` | Open new tab |
| `goBack()` | Navigate back |
| `selectAll()` | Ctrl/Cmd+A |
| `waitForStability(timeout?)` | Wait for page to stabilize |
| `retrieveTabState()` | Get current tab info |

### Image

Screenshot results are `Image` objects:

```typescript
const screenshot = await harness.screenshot();

// Get dimensions
const { width, height } = await screenshot.getDimensions();

// Get base64
const base64 = await screenshot.toBase64();

// Get buffer
const buffer = await screenshot.toBuffer();

// Resize
const resized = await screenshot.resize(800, 600);

// Save to file
await screenshot.saveToFile('output.png');
```

### BrowserProvider

```typescript
const provider = BrowserProvider.getInstance();

// Default options
const context = await provider.newContext();

// Custom launch options
const context = await provider.newContext({
  launchOptions: {
    headless: true,
    args: ['--no-sandbox']
  },
  contextOptions: {
    viewport: { width: 1920, height: 1080 }
  }
});

// Connect via CDP
const context = await provider.newContext({
  cdp: 'http://localhost:9222'
});

// Use existing browser instance
const context = await provider.newContext({
  instance: existingBrowser
});
```

## CLI

```bash
# Demo with visual feedback
magnitude-harness demo --url https://example.com

# Capture screenshot
magnitude-harness navigate https://example.com --output screenshot.png --headless

# Interactive session
magnitude-harness interactive --url https://google.com
```

## License

Apache-2.0
