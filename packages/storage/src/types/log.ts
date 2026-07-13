export interface StoredLogEntry {
  readonly level: 'DEBUG' | 'INFO' | 'WARN' | 'ERROR'
  readonly timestamp: string
  readonly msg?: string
  readonly [key: string]: unknown
}