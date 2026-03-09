# Chat Persistence Module

This module provides the infrastructure for persisting Magnitude chat sessions to disk.

## Architecture

The persistence layer follows Sage's proven patterns, adapted for Magnitude's JSON file storage:

- **Contract-based design**: `ChatPersistenceService` interface defines the API
- **Multiple implementations**: JSON file storage for production, in-memory for testing
- **Atomic writes**: Uses temp file + rename pattern for safe concurrent access
- **Session isolation**: Each chat session stored in separate file with cuid2 ID

## Files

- `chat-persistence.ts` - Service interface and types
- `json-file-persistence.ts` - Production implementation (JSON files)
- `in-memory-persistence.ts` - Test implementation (in-memory)
- `persistence.test.ts` - Comprehensive test suite
- `index.ts` - Public exports

## Usage

### Creating a persistence service

```typescript
import { createJsonFilePersistence } from './persistence'
import { createId } from '@paralleldrive/cuid2'

const sessionId = createId()
const persistence = createJsonFilePersistence(
  sessionId,
  process.cwd(),
  'main' // git branch
)
```

### Persisting events

```typescript
import type { AppEvent } from '@magnitudedev/agent/events'

const events: AppEvent[] = [
  {
    type: 'session_initialized',
    forkId: null,
    context: { /* ... */ }
  },
  // ... more events
]

await persistence.persistNewEvents(events)
```

### Loading events (hydration)

```typescript
const events = await persistence.loadEvents()
// Returns all events in order they were persisted
```

### Managing metadata

```typescript
// Get metadata
const metadata = await persistence.getSessionMetadata()
console.log(metadata.chatName, metadata.created, metadata.updated)

// Update chat name
await persistence.saveSessionMetadata({ chatName: 'My Chat' })
```

## Storage Format

Sessions are stored in `~/.magnitude/sessions/<sessionId>.json`:

```json
{
  "sessionId": "ckx7y8z9...",
  "created": "2026-02-10T19:00:00.000Z",
  "updated": "2026-02-10T19:05:00.000Z",
  "metadata": {
    "chatName": "Implement chat persistence",
    "workingDirectory": "/Users/anerli/magnitude",
    "gitBranch": "main"
  },
  "events": [
    {
      "type": "session_initialized",
      "forkId": null,
      "context": { /* ... */ },
      "timestamp": 1707595200000
    },
    // ... more events
  ]
}
```

## Testing

Run tests:
```bash
cd cli
bun test src/persistence/persistence.test.ts
```

For testing without file I/O:
```typescript
import { createInMemoryPersistence } from './persistence'

const persistence = createInMemoryPersistence(sessionId, cwd, branch)
// Same API, but stores everything in memory
```

## Integration Points

This module is designed to integrate with:

1. **EventSink** (event-core) - Provides events to persist
2. **LifecycleCoordinator** (agent) - Triggers persistence on STABLE state
3. **Session initialization** - Loads events for hydration
4. **HydrationContext** (event-core) - Prevents duplicate persistence during replay

## Error Handling

All methods can throw `PersistenceError`:

```typescript
type PersistenceError =
  | { type: 'load_failed'; message: string }
  | { type: 'save_failed'; message: string }
  | { type: 'file_error'; message: string }
```

The calling code should handle these errors appropriately (retry, log, notify user, etc.).

## Future Enhancements

- Event compaction (remove old events after sandbox_reset)
- Session listing and management UI
- Compression for large sessions
- Backup/restore functionality
- Cloud sync support
