//! Engine telemetry: OpenTelemetry spans exported over OTLP-HTTP.
//!
//! `Telemetry::open` installs the process-global tracer provider (default
//! endpoint: the local Motel collector's standard `/v1/traces` path). Every
//! engine phase then records spans through the global tracer — compilation
//! (entry, kernel count, estimated cost, optimality), weight import (name,
//! encoding, elements), and one child span per executed launch (dispatched
//! geometry and wall time, from the executor's own `LaunchExecution`
//! records). With no provider installed the global tracer is a no-op, so
//! telemetry never breaks execution; export failures are ignored.

use opentelemetry::global;
use opentelemetry::trace::{Span, SpanKind, Tracer};
use opentelemetry::KeyValue;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::trace::SdkTracerProvider;

/// The default OTLP-HTTP traces endpoint (the local Motel collector).
pub const DEFAULT_TRACES_ENDPOINT: &str = "http://127.0.0.1:27686/v1/traces";

const SERVICE: &str = "magnitude-engine";

/// The installed global telemetry: flush on shutdown.
pub struct Telemetry {
    provider: SdkTracerProvider,
}

impl Telemetry {
    /// Install the global OTLP-HTTP tracer provider for this process.
    pub fn open(endpoint: &str) -> Telemetry {
        let provider = match opentelemetry_otlp::SpanExporter::builder()
            .with_http()
            .with_protocol(opentelemetry_otlp::Protocol::HttpJson)
            .with_endpoint(endpoint)
            .build()
        {
            Ok(exporter) => SdkTracerProvider::builder()
                .with_span_processor(opentelemetry_sdk::trace::SimpleSpanProcessor::new(
                    exporter,
                ))
                .build(),
            Err(_) => SdkTracerProvider::builder().build(),
        };
        global::set_tracer_provider(provider.clone());
        Telemetry { provider }
    }

    /// Flush everything exported so far. Errors are ignored.
    pub fn flush(&self) {
        let _ = self.provider.force_flush();
    }
}

fn key_u64(key: &str, value: u64) -> KeyValue {
    KeyValue::new(
        key.to_string(),
        opentelemetry::Value::I64(i64::try_from(value).unwrap_or(i64::MAX)),
    )
}

fn key_str(key: &str, value: impl Into<String>) -> KeyValue {
    KeyValue::new(
        key.to_string(),
        opentelemetry::Value::String(value.into().into()),
    )
}

/// The engine's tracer (a no-op tracer until `Telemetry::open` ran).
pub fn tracer() -> global::BoxedTracer {
    global::tracer(SERVICE)
}

/// Record one compiled entry: entry name, kernel count, the solver's
/// estimated cost and optimality verdict, and wall time.
pub fn span_compile(
    entry: &str,
    kernels: u64,
    estimated_cost: u64,
    optimal: bool,
    seconds: f64,
) {
    let tracer = tracer();
    let mut span = tracer
        .span_builder(format!("compile {entry}"))
        .with_kind(SpanKind::Internal)
        .with_attributes(vec![
            key_str("magnitude.entry", entry),
            key_u64("magnitude.kernels", kernels),
            key_u64("magnitude.estimated_cost", estimated_cost),
            KeyValue::new("magnitude.optimal", optimal),
            KeyValue::new("magnitude.seconds", seconds),
        ])
        .start(&tracer);
    span.end();
}

/// Record one imported weight (name, encoding, elements, wall time).
pub fn span_import(weight: &str, encoding: &str, elements: u64, seconds: f64) {
    let tracer = tracer();
    let mut span = tracer
        .span_builder(format!("import {weight}"))
        .with_kind(SpanKind::Internal)
        .with_attributes(vec![
            key_str("magnitude.weight", weight),
            key_str("magnitude.encoding", encoding),
            key_u64("magnitude.elements", elements),
            KeyValue::new("magnitude.seconds", seconds),
        ])
        .start(&tracer);
    span.end();
}

/// Record one executed launch: dispatched geometry and wall time.
pub fn span_launch(launch: &seismic_realization::physical::LaunchExecution) {
    let tracer = tracer();
    let mut span = tracer
        .span_builder(format!("launch {}", launch.launch.index()))
        .with_kind(SpanKind::Internal)
        .with_attributes(vec![
            key_u64("magnitude.work_items", launch.work_items),
            key_u64("magnitude.participants", launch.participants),
            key_u64("magnitude.workgroups.0", launch.workgroups[0]),
            key_u64("magnitude.workgroups.1", launch.workgroups[1]),
            key_u64("magnitude.workgroups.2", launch.workgroups[2]),
            KeyValue::new("magnitude.seconds", launch.seconds),
        ])
        .start(&tracer);
    span.end();
}
