use std::time::{Duration, SystemTime};

use anyhow::Context as _;
use axum::extract::MatchedPath;
use axum::http::{HeaderMap, Request};
use opentelemetry::global;
use opentelemetry::logs::LogRecord as _;
use opentelemetry::propagation::Extractor;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry::{InstrumentationScope, KeyValue};
use opentelemetry_appender_tracing::layer::OpenTelemetryTracingBridge;
use opentelemetry_otlp::{Protocol, WithExportConfig};
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::error::OTelSdkResult;
use opentelemetry_sdk::logs::{
    BatchConfigBuilder as LogBatchConfigBuilder, BatchLogProcessor, LogProcessor, SdkLogRecord,
    SdkLoggerProvider,
};
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::{
    BatchConfigBuilder as TraceBatchConfigBuilder, BatchSpanProcessor, SdkTracerProvider,
};
use tracing::Span;
use tracing_opentelemetry::OpenTelemetrySpanExt as _;
use tracing_subscriber::Layer as _;
use tracing_subscriber::filter::EnvFilter;
use tracing_subscriber::layer::SubscriberExt as _;
use tracing_subscriber::util::SubscriberInitExt as _;

pub const SERVICE_NAME: &str = "magnitude-icn";
const MOTEL_BASE_URL: &str = "http://127.0.0.1:27686";
const EXPORT_INTERVAL: Duration = Duration::from_secs(1);

pub struct TelemetryGuard {
    logger_provider: Option<SdkLoggerProvider>,
    tracer_provider: Option<SdkTracerProvider>,
}

impl Drop for TelemetryGuard {
    fn drop(&mut self) {
        if let Some(provider) = self.logger_provider.take() {
            let _ = provider.shutdown();
        }
        if let Some(provider) = self.tracer_provider.take() {
            let _ = provider.shutdown();
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct OtlpEndpoints {
    traces: String,
    logs: String,
}

pub fn init() -> anyhow::Result<TelemetryGuard> {
    global::set_text_map_propagator(TraceContextPropagator::new());

    let Some(endpoints) = otlp_endpoints() else {
        tracing_subscriber::registry()
            .with(stderr_layer())
            .try_init()
            .context("failed to install ICN tracing subscriber")?;
        return Ok(TelemetryGuard {
            logger_provider: None,
            tracer_provider: None,
        });
    };

    let resource = Resource::builder()
        .with_service_name(SERVICE_NAME)
        .with_attributes([
            KeyValue::new("service.version", env!("CARGO_PKG_VERSION")),
            KeyValue::new("process.executable.name", "magnitude-icn"),
        ])
        .build();

    let trace_exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_http()
        .with_protocol(Protocol::HttpJson)
        .with_endpoint(endpoints.traces)
        .build()
        .context("failed to create ICN OTLP trace exporter")?;
    let trace_processor = BatchSpanProcessor::builder(trace_exporter)
        .with_batch_config(
            TraceBatchConfigBuilder::default()
                .with_scheduled_delay(EXPORT_INTERVAL)
                .build(),
        )
        .build();
    let tracer_provider = SdkTracerProvider::builder()
        .with_resource(resource.clone())
        .with_span_processor(trace_processor)
        .build();
    let tracer = tracer_provider.tracer(SERVICE_NAME);

    let log_exporter = opentelemetry_otlp::LogExporter::builder()
        .with_http()
        .with_protocol(Protocol::HttpJson)
        .with_endpoint(endpoints.logs)
        .build()
        .context("failed to create ICN OTLP log exporter")?;
    let log_processor = BatchLogProcessor::builder(log_exporter)
        .with_batch_config(
            LogBatchConfigBuilder::default()
                .with_scheduled_delay(EXPORT_INTERVAL)
                .build(),
        )
        .build();
    let logger_provider = SdkLoggerProvider::builder()
        .with_resource(resource)
        .with_log_processor(EventTimestampProcessor)
        .with_log_processor(log_processor)
        .build();

    let trace_layer = tracing_opentelemetry::layer()
        .with_tracer(tracer)
        .with_filter(otel_filter());
    let log_layer = OpenTelemetryTracingBridge::new(&logger_provider).with_filter(otel_filter());

    tracing_subscriber::registry()
        .with(stderr_layer())
        .with(trace_layer)
        .with(log_layer)
        .try_init()
        .context("failed to install ICN OpenTelemetry subscriber")?;

    tracing::info!(
        service.name = SERVICE_NAME,
        otel.traces.endpoint = %otlp_endpoints().expect("OTLP was configured").traces,
        otel.logs.endpoint = %otlp_endpoints().expect("OTLP was configured").logs,
        "OpenTelemetry export enabled"
    );

    Ok(TelemetryGuard {
        logger_provider: Some(logger_provider),
        tracer_provider: Some(tracer_provider),
    })
}

/// `tracing` events have an observation time but no event time in the upstream
/// bridge. Motel indexes OTLP logs by event time, so populate it before the
/// batching processor snapshots the record.
#[derive(Debug)]
struct EventTimestampProcessor;

impl LogProcessor for EventTimestampProcessor {
    fn emit(&self, record: &mut SdkLogRecord, _instrumentation: &InstrumentationScope) {
        if record.timestamp().is_none() {
            record.set_timestamp(SystemTime::now());
        }
    }

    fn force_flush(&self) -> OTelSdkResult {
        Ok(())
    }

    fn shutdown_with_timeout(&self, _timeout: Duration) -> OTelSdkResult {
        Ok(())
    }
}

pub fn http_request_span<B>(request: &Request<B>) -> Span {
    let route = request
        .extensions()
        .get::<MatchedPath>()
        .map_or_else(|| request.uri().path(), MatchedPath::as_str);
    let span = tracing::info_span!(
        "http.request",
        http.request.method = %request.method(),
        http.route = route,
        url.path = request.uri().path(),
        network.protocol.version = ?request.version(),
    );
    let parent = global::get_text_map_propagator(|propagator| {
        propagator.extract(&HeaderExtractor(request.headers()))
    });
    let _ = span.set_parent(parent);
    span
}

fn stderr_layer<S>() -> impl tracing_subscriber::Layer<S>
where
    S: tracing::Subscriber + for<'span> tracing_subscriber::registry::LookupSpan<'span>,
{
    tracing_subscriber::fmt::layer()
        .with_writer(std::io::stderr)
        .with_thread_names(true)
        .with_filter(runtime_filter())
}

fn runtime_filter() -> EnvFilter {
    EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"))
}

fn otel_filter() -> EnvFilter {
    runtime_filter()
        .add_directive("opentelemetry=off".parse().expect("valid directive"))
        .add_directive("reqwest=off".parse().expect("valid directive"))
        .add_directive("hyper=off".parse().expect("valid directive"))
}

fn otlp_endpoints() -> Option<OtlpEndpoints> {
    let magnitude_base = std::env::var("MAGNITUDE_OTEL_ENDPOINT").ok();
    let standard_base = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT").ok();
    let default_base =
        (std::env::var("MAGNITUDE_OTEL").as_deref() == Ok("1")).then(|| MOTEL_BASE_URL.to_owned());
    let base = magnitude_base.or(standard_base).or(default_base);
    let trace_endpoint = std::env::var("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT").ok();
    let log_endpoint = std::env::var("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT").ok();

    let inferred_base = base.or_else(|| {
        trace_endpoint
            .as_deref()
            .and_then(|endpoint| endpoint.strip_suffix("/v1/traces"))
            .or_else(|| {
                log_endpoint
                    .as_deref()
                    .and_then(|endpoint| endpoint.strip_suffix("/v1/logs"))
            })
            .map(str::to_owned)
    });
    let traces = trace_endpoint.or_else(|| {
        inferred_base
            .as_deref()
            .map(|base| signal_endpoint(base, "traces"))
    })?;
    let logs = log_endpoint.or_else(|| {
        inferred_base
            .as_deref()
            .map(|base| signal_endpoint(base, "logs"))
    })?;
    Some(OtlpEndpoints { traces, logs })
}

fn signal_endpoint(base: &str, signal: &str) -> String {
    let base = base
        .trim_end_matches('/')
        .strip_suffix("/v1/traces")
        .or_else(|| base.trim_end_matches('/').strip_suffix("/v1/logs"))
        .unwrap_or_else(|| base.trim_end_matches('/'));
    format!("{base}/v1/{signal}")
}

struct HeaderExtractor<'a>(&'a HeaderMap);

impl Extractor for HeaderExtractor<'_> {
    fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).and_then(|value| value.to_str().ok())
    }

    fn keys(&self) -> Vec<&str> {
        self.0.keys().map(|name| name.as_str()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signal_endpoints_accept_a_base_or_signal_url() {
        assert_eq!(
            signal_endpoint("http://127.0.0.1:27686", "traces"),
            "http://127.0.0.1:27686/v1/traces"
        );
        assert_eq!(
            signal_endpoint("http://127.0.0.1:27686/v1/traces", "logs"),
            "http://127.0.0.1:27686/v1/logs"
        );
    }
}
