use std::borrow::Cow;
use std::io::Write;
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
use opentelemetry_otlp::{Protocol, WithExportConfig, WithHttpConfig};
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
use tracing_subscriber::fmt::MakeWriter;
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

pub fn init(export: bool) -> anyhow::Result<TelemetryGuard> {
    global::set_text_map_propagator(TraceContextPropagator::new());

    let Some(endpoints) = export.then(otlp_endpoints).flatten() else {
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
            KeyValue::new("process.executable.name", env!("CARGO_BIN_NAME")),
        ])
        .build();

    let trace_exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_http()
        .with_http_client(RedactingHttpClient::new())
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
        .with_http_client(RedactingHttpClient::new())
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

/// The engine worker's subscriber: events to standard error only, without colour, since
/// the service records each line in its own log.
pub fn init_worker() {
    let _ = tracing_subscriber::fmt()
        .with_writer(Redacted(std::io::stderr))
        .with_ansi(false)
        .with_env_filter(runtime_filter())
        .try_init();
}

fn stderr_layer<S>() -> impl tracing_subscriber::Layer<S>
where
    S: tracing::Subscriber + for<'span> tracing_subscriber::registry::LookupSpan<'span>,
{
    tracing_subscriber::fmt::layer()
        .with_writer(Redacted(std::io::stderr))
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

/// Masks credentials in log text: a Hugging Face token (`hf_` and at least
/// [`MIN_HUB_TOKEN_TAIL`] token characters) becomes `hf_***`, and any Bearer credential (an
/// `Authorization` header value among them) becomes `Bearer ***`.
fn redact(text: &str) -> Cow<'_, str> {
    let bytes = text.as_bytes();
    let mut redacted: Option<String> = None;
    let mut copied = 0;
    let mut index = 0;
    while index < bytes.len() {
        let Some((mask, end)) = credential_at(bytes, index) else {
            index += 1;
            continue;
        };
        let redacted = redacted.get_or_insert_with(|| String::with_capacity(text.len()));
        // Both bounds sit on ASCII bytes, so on character boundaries.
        redacted.push_str(&text[copied..index]);
        redacted.push_str(mask);
        index = end;
        copied = end;
    }
    match redacted {
        Some(mut redacted) => {
            redacted.push_str(&text[copied..]);
            Cow::Owned(redacted)
        }
        None => Cow::Borrowed(text),
    }
}

/// Shorter `hf_` words (`hf_hub` in a module path) are not tokens; real ones are far longer.
const MIN_HUB_TOKEN_TAIL: usize = 8;
const BEARER: &[u8] = b"bearer ";

fn credential_at(bytes: &[u8], index: usize) -> Option<(&'static str, usize)> {
    let rest = &bytes[index..];
    if rest.starts_with(b"hf_") {
        let tail = rest[3..]
            .iter()
            .take_while(|byte| byte.is_ascii_alphanumeric() || **byte == b'_')
            .count();
        return (tail >= MIN_HUB_TOKEN_TAIL).then_some(("hf_***", index + 3 + tail));
    }
    let word_start = index == 0 || !bytes[index - 1].is_ascii_alphanumeric();
    if word_start && rest.len() > BEARER.len() && rest[..BEARER.len()].eq_ignore_ascii_case(BEARER)
    {
        let value = &rest[BEARER.len()..];
        let length = value
            .iter()
            .take_while(|byte| {
                !byte.is_ascii_whitespace() && !matches!(byte, b'"' | b'\'' | b'\\' | b',' | b';')
            })
            .count();
        if length > 0 && &value[..length] != b"***" {
            return Some(("Bearer ***", index + BEARER.len() + length));
        }
    }
    None
}

/// A writer factory whose writers pass every line through [`redact`]. The fmt layer writes each
/// event whole, so no credential spans two writes.
#[derive(Clone, Copy)]
struct Redacted<M>(M);

impl<'a, M: MakeWriter<'a>> MakeWriter<'a> for Redacted<M> {
    type Writer = RedactingWriter<M::Writer>;

    fn make_writer(&'a self) -> Self::Writer {
        RedactingWriter(self.0.make_writer())
    }

    fn make_writer_for(&'a self, meta: &tracing::Metadata<'_>) -> Self::Writer {
        RedactingWriter(self.0.make_writer_for(meta))
    }
}

struct RedactingWriter<W>(W);

impl<W: Write> Write for RedactingWriter<W> {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        let text = String::from_utf8_lossy(buffer);
        self.0.write_all(redact(&text).as_bytes())?;
        Ok(buffer.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.0.flush()
    }
}

/// The OTLP exporters' HTTP client: the blocking reqwest client they use by default, with each
/// exported JSON payload (spans, logs) passed through [`redact`]. A mask keeps the JSON valid: a
/// token never contains a quote or a backslash.
#[derive(Debug)]
struct RedactingHttpClient(reqwest::blocking::Client);

impl RedactingHttpClient {
    fn new() -> Self {
        // The blocking client can't be built on a runtime thread.
        Self(
            std::thread::spawn(|| {
                reqwest::blocking::Client::builder()
                    .timeout(Duration::from_secs(10))
                    .build()
                    .unwrap_or_else(|_| reqwest::blocking::Client::new())
            })
            .join()
            .unwrap_or_else(|_| reqwest::blocking::Client::new()),
        )
    }
}

#[async_trait::async_trait]
impl opentelemetry_http::HttpClient for RedactingHttpClient {
    async fn send_bytes(
        &self,
        request: opentelemetry_http::Request<opentelemetry_http::Bytes>,
    ) -> Result<opentelemetry_http::Response<opentelemetry_http::Bytes>, opentelemetry_http::HttpError>
    {
        let request = request.map(|body| match std::str::from_utf8(&body) {
            Ok(text) => match redact(text) {
                Cow::Owned(redacted) => opentelemetry_http::Bytes::from(redacted),
                Cow::Borrowed(_) => body,
            },
            Err(_) => body,
        });
        self.0.send_bytes(request).await
    }
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

    const TOKEN: &str = "hf_AbCdEfGhIjKlMnOpQrStUvWxYz012345678";

    #[test]
    fn hugging_face_tokens_and_bearer_credentials_are_masked() {
        for (text, expected) in [
            (format!("token {TOKEN} rejected"), "token hf_*** rejected".to_owned()),
            (format!("Authorization: Bearer {TOKEN}"), "Authorization: Bearer ***".to_owned()),
            (
                "authorization: bearer private-capability\n".to_owned(),
                "authorization: Bearer ***\n".to_owned(),
            ),
            (
                format!(r#"{{"authorization": "Bearer {TOKEN}", "x": 1}}"#),
                r#"{"authorization": "Bearer ***", "x": 1}"#.to_owned(),
            ),
            (format!("é{TOKEN}é"), "éhf_***é".to_owned()),
            ("hf_hub::client: no token found".to_owned(), "hf_hub::client: no token found".to_owned()),
            ("Bearer ***".to_owned(), "Bearer ***".to_owned()),
            ("a Bearer".to_owned(), "a Bearer".to_owned()),
        ] {
            assert_eq!(redact(&text), expected);
        }
        assert!(matches!(redact("nothing secret"), Cow::Borrowed(_)));
    }

    #[derive(Clone, Default)]
    struct Captured(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

    impl Write for Captured {
        fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buffer);
            Ok(buffer.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    struct NoFingerprints;

    impl magnitude_service_models::CatalogFingerprints for NoFingerprints {
        fn catalog_fingerprints(
            &self,
        ) -> futures_util::future::BoxFuture<
            '_,
            Vec<(
                magnitude_service_contracts::models::ModelId,
                magnitude_service_contracts::models::ModelFingerprint,
            )>,
        > {
            Box::pin(async { Vec::new() })
        }
    }

    /// Every Hub request presents the configured token, and the Hub rejects it here, so the
    /// token passes through every request and fallback path while everything is logged.
    #[tokio::test]
    async fn a_configured_token_never_reaches_the_log() {
        use magnitude_service_contracts::{HuggingFaceCredential, HuggingFaceModelCatalog as _};
        use std::io::{BufRead, BufReader};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        std::thread::spawn(move || {
            for connection in listener.incoming() {
                let Ok(mut connection) = connection else {
                    return;
                };
                let mut reader = BufReader::new(connection.try_clone().unwrap());
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                        break;
                    }
                }
                let _ = write!(
                    connection,
                    "HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                );
            }
        });
        let captured = Captured::default();
        let writer = captured.clone();
        let subscriber = tracing_subscriber::registry().with(
            tracing_subscriber::fmt::layer()
                .with_ansi(false)
                .with_writer(Redacted(move || writer.clone()))
                .with_filter(tracing_subscriber::filter::LevelFilter::TRACE),
        );
        let _guard = tracing::subscriber::set_default(subscriber);

        let temporary = tempfile::tempdir().unwrap();
        let mut config = magnitude_service_models::InventoryConfig::with_roots(
            temporary.path().join("model-store"),
            temporary.path().join("cache"),
        )
        .unwrap();
        config.local_models.roots.clear();
        config.hub_endpoint = Some(endpoint);
        let store = std::sync::Arc::new(
            magnitude_service_models::ManagedModelStore::open(config)
                .await
                .unwrap(),
        );
        let discovery = magnitude_service_models::HuggingFaceDiscovery::new(
            store.clone(),
            magnitude_service_models::candidate_files::RecognizedArchitectures::default(),
            magnitude_service_models::lineage::CatalogLineage::new(
                store.clone(),
                magnitude_service_contracts::models::RecommendableModelCatalog {
                    models: Vec::new(),
                    diagnostics: Vec::new(),
                },
            ),
            std::sync::Arc::new(NoFingerprints),
        );
        discovery
            .configure_credential(HuggingFaceCredential {
                token: Some(TOKEN.to_owned()),
            })
            .await
            .unwrap();
        tracing::info!("presenting Authorization: Bearer {TOKEN}");
        let search = discovery
            .search(magnitude_service_contracts::HuggingFaceModelSearchRequest {
                query: "model".to_owned(),
                limit: 5,
            })
            .await;
        tracing::warn!(?search, "search");
        let resolve = discovery
            .resolve(magnitude_service_contracts::HuggingFaceRepositoryRequest {
                repository: "owner/model".to_owned(),
                revision: "main".to_owned(),
            })
            .await;
        tracing::warn!(?resolve, "resolve");

        let log = String::from_utf8(captured.0.lock().unwrap().clone()).unwrap();
        assert!(log.contains("Bearer ***"), "{log}");
        assert!(!log.contains(&TOKEN[3..]), "{log}");
    }

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
