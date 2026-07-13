# CodingAgent Projection Dependency Graph

```mermaid
flowchart TD

    SessionContext[SessionContext]
    WorkingState([WorkingState])
    Turn[Turn]
    Fork[Fork]
    Memory([Memory])
    Display([Display])

    SessionContext --> Memory
    Fork -.->|completed| WorkingState
    Fork -.->|created| Memory
    Fork -.->|completed| Memory
    Fork -.->|completed| Display
```

## Legend

- `[Name]` - Standard projection
- `([Name])` - Forked projection (per-fork state)
- `A --> B` - B reads from A
- `A -.->|signal| B` - B subscribes to signal from A
