//! OpenTelemetry at the runtime's public boundaries (spec §15.3):
//! preparation, selection, allocation, execution. Observational only; no
//! attribute feeds back into planning. With no global provider installed
//! every call here is a no-op.

use opentelemetry::global::{self, BoxedSpan, BoxedTracer};
use opentelemetry::metrics::{Counter, Histogram};
use opentelemetry::trace::{Span, SpanKind, Tracer};
use opentelemetry::KeyValue;
use std::sync::OnceLock;
use std::time::Instant;

const SCOPE: &str = "seismic";

fn tracer() -> BoxedTracer {
    global::tracer(SCOPE)
}

pub(crate) fn key_u64(key: impl Into<opentelemetry::Key>, value: u64) -> KeyValue {
    KeyValue::new(
        key,
        opentelemetry::Value::I64(i64::try_from(value).unwrap_or(i64::MAX)),
    )
}

pub(crate) fn key_str(key: &'static str, value: impl Into<String>) -> KeyValue {
    KeyValue::new(key, opentelemetry::Value::String(value.into().into()))
}

pub(crate) fn key_bool(key: &'static str, value: bool) -> KeyValue {
    KeyValue::new(key, value)
}

pub(crate) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// One timed span. Ended on drop with its elapsed milliseconds attached.
pub(crate) struct Timed {
    span: BoxedSpan,
    started: Instant,
}

impl Timed {
    pub(crate) fn start(name: &'static str, attributes: Vec<KeyValue>) -> Self {
        let tracer = tracer();
        let span = tracer
            .span_builder(name)
            .with_kind(SpanKind::Internal)
            .with_attributes(attributes)
            .start(&tracer);
        Self {
            span,
            started: Instant::now(),
        }
    }

    pub(crate) fn attribute(&mut self, attribute: KeyValue) {
        self.span.set_attribute(attribute);
    }

    pub(crate) fn elapsed_ms(&self) -> f64 {
        self.started.elapsed().as_secs_f64() * 1e3
    }
}

impl Drop for Timed {
    fn drop(&mut self) {
        self.span
            .set_attribute(KeyValue::new("seismic.elapsed_ms", self.elapsed_ms()));
        self.span.end();
    }
}

struct Instruments {
    preparations: Counter<u64>,
    preparation_ms: Histogram<f64>,
    calls: Counter<u64>,
    call_ms: Histogram<f64>,
    allocated_bytes: Histogram<u64>,
}

fn instruments() -> &'static Instruments {
    static INSTRUMENTS: OnceLock<Instruments> = OnceLock::new();
    INSTRUMENTS.get_or_init(|| {
        let meter = global::meter(SCOPE);
        Instruments {
            preparations: meter.u64_counter("seismic.preparations").build(),
            preparation_ms: meter
                .f64_histogram("seismic.preparation.duration_ms")
                .build(),
            calls: meter.u64_counter("seismic.calls").build(),
            call_ms: meter.f64_histogram("seismic.call.duration_ms").build(),
            allocated_bytes: meter.u64_histogram("seismic.call.allocated_bytes").build(),
        }
    })
}

pub(crate) fn record_preparation(elapsed_ms: f64, attributes: &[KeyValue]) {
    let instruments = instruments();
    instruments.preparations.add(1, attributes);
    instruments.preparation_ms.record(elapsed_ms, attributes);
}

pub(crate) fn record_call(elapsed_ms: f64, allocated_bytes: u64, attributes: &[KeyValue]) {
    let instruments = instruments();
    instruments.calls.add(1, attributes);
    instruments.call_ms.record(elapsed_ms, attributes);
    instruments
        .allocated_bytes
        .record(allocated_bytes, attributes);
}
