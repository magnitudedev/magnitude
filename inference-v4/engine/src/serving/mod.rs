//! HTTP adapts the shared service. Run on a Tokio LocalSet: native templates and
//! parsers remain host-local while the numerical worker owns model execution.
pub mod startup;

use crate::{
    chat::{
        wire::{ModelLimits, Request as WireRequest},
        CompleteResponse, Session, SseResponse, TemplateBundle, TemplateSelection, TemplateVariant,
    },
    inputs::ByteBpeTokenizer,
    service::{
        owner::Executor,
        runtime::{Client, Service},
    },
};
use bytes::Bytes;
use hyper::{
    body::{Body, Frame, Incoming, SizeHint},
    server::conn::http1,
    service::service_fn,
    Method, Request, Response, StatusCode,
};
use hyper_util::rt::{TokioIo, TokioTimer};
use serde_json::json;
use std::{
    convert::Infallible,
    future::{poll_fn, Future},
    pin::Pin,
    rc::Rc,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    task::{Context, Poll},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::{net::TcpListener, sync::mpsc, task::JoinSet};

pub struct Config {
    pub model: String,
    pub context_tokens: usize,
    pub vocabulary: usize,
    pub output_capacity: usize,
    pub forced_quantum: usize,
    pub template_variant: Option<String>,
    pub template_override: Option<TemplateVariant>,
    pub max_body_bytes: usize,
    pub max_response_bytes: usize,
    pub max_connections: usize,
    pub request_timeout: Duration,
}
impl Config {
    fn validate(
        &self,
        tokenizer: &ByteBpeTokenizer,
        templates: &TemplateBundle,
    ) -> Result<(), String> {
        if self.model.is_empty()
            || self.model.len() > 1024
            || self.context_tokens == 0
            || self.context_tokens > i32::MAX as usize
            || self.vocabulary < tokenizer.vocabulary()
            || self.output_capacity == 0
            || self.max_body_bytes == 0
            || self.max_response_bytes < 256
            || self.max_connections == 0
            || self.request_timeout.is_zero()
            || (self.template_variant.is_some() && self.template_override.is_some())
        {
            return Err("invalid HTTP model or resource limits".into());
        }
        templates.select(
            false,
            &TemplateSelection {
                variant: self.template_variant.as_deref(),
                source_override: self.template_override.as_ref(),
            },
        )?;
        Ok(())
    }
}
struct Host<E: Executor + 'static> {
    client: Client<E>,
    tokenizer: Arc<ByteBpeTokenizer>,
    templates: TemplateBundle,
    config: Config,
    identity: String,
    next: std::cell::Cell<u64>,
}
pub struct Server<E: Executor + 'static> {
    service: Service<E>,
    host: Rc<Host<E>>,
}
impl<E: Executor + 'static> Server<E> {
    pub fn new(
        service: Service<E>,
        tokenizer: Arc<ByteBpeTokenizer>,
        templates: TemplateBundle,
        config: Config,
    ) -> Result<Self, String> {
        config.validate(&tokenizer, &templates)?;
        static NEXT_SERVER: AtomicU64 = AtomicU64::new(0);
        let server = NEXT_SERVER
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| "server identity exhausted")?;
        let time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| error.to_string())?;
        let host = Rc::new(Host {
            client: service.client(),
            tokenizer,
            templates,
            config,
            identity: format!("{:x}-{}-{server}", time.as_nanos(), std::process::id()),
            next: std::cell::Cell::new(0),
        });
        Ok(Self { service, host })
    }
    /// The caller supplies the listener and shutdown signal. Shutdown drops
    /// connection/session futures before joining the numerical worker, which
    /// retains outstanding native work until completion.
    pub async fn serve(
        mut self,
        listener: TcpListener,
        shutdown: impl Future<Output = ()>,
    ) -> Result<(), String> {
        let mut connections = JoinSet::new();
        tokio::pin!(shutdown);
        let result = loop {
            tokio::select! {
                _ = &mut shutdown => break Ok(()),
                accepted = listener.accept(), if connections.len() < self.host.config.max_connections => {
                    match accepted {
                        Ok((socket, _)) => {
                            let host = self.host.clone();
                            connections.spawn_local(async move {
                                let timeout = host.config.request_timeout;
                                let service = service_fn(move |request| {
                                    let host = host.clone();
                                    async move { Ok::<_, Infallible>(host.route(request).await) }
                                });
                                // Connection failure/disconnect drops the handler
                                // and response body, including any native session.
                                let _ = http1::Builder::new().timer(TokioTimer::new())
                                    .header_read_timeout(timeout)
                                    .serve_connection(TokioIo::new(socket), service).await;
                            });
                        }
                        Err(error) => break Err(error.to_string()),
                    }
                }
                _ = connections.join_next(), if !connections.is_empty() => {},
            }
        };
        drop(listener);
        connections.abort_all();
        while connections.join_next().await.is_some() {}
        self.service.close();
        result
    }
}
impl<E: Executor + 'static> Host<E> {
    async fn route(self: Rc<Self>, request: Request<Incoming>) -> Response<ResponseBody> {
        let limit = self.config.max_response_bytes;
        let response = self.route_inner(request).await;
        if matches!(response.body(), ResponseBody::Full(Some(bytes)) if bytes.len() > limit) {
            let status = if response.status().is_success() {
                StatusCode::INTERNAL_SERVER_ERROR
            } else {
                response.status()
            };
            return error_response(status, "response byte limit exceeded", "server_error");
        }
        response
    }
    async fn route_inner(self: Rc<Self>, request: Request<Incoming>) -> Response<ResponseBody> {
        match (request.method(), request.uri().path()) {
            (&Method::GET, "/health") => {
                return match self.client.check().await {
                    Ok(()) => json_response(
                        StatusCode::OK,
                        json!({"model":self.config.model,
                    "context_tokens":self.config.context_tokens,"vocabulary":self.config.vocabulary,"ready":true}),
                    ),
                    Err(error) => {
                        error_response(StatusCode::SERVICE_UNAVAILABLE, &error, "server_error")
                    }
                }
            }
            (&Method::GET, "/v1/models") => {
                return json_response(
                    StatusCode::OK,
                    json!({"object":"list","data":[{
                "id":self.config.model,"object":"model","created":0,"owned_by":"magnitude"}]}),
                )
            }
            (&Method::POST, "/v1/chat/completions") => {}
            (_, "/health" | "/v1/models" | "/v1/chat/completions") => {
                return error_response(
                    StatusCode::METHOD_NOT_ALLOWED,
                    "method not allowed",
                    "invalid_request_error",
                )
            }
            _ => {
                return error_response(
                    StatusCode::NOT_FOUND,
                    "route not found",
                    "invalid_request_error",
                )
            }
        }
        let bytes = match tokio::time::timeout(
            self.config.request_timeout,
            read_body(request.into_body(), self.config.max_body_bytes),
        )
        .await
        {
            Ok(Ok(bytes)) => bytes,
            Ok(Err((status, error))) => {
                return error_response(status, &error, "invalid_request_error")
            }
            Err(_) => {
                return error_response(
                    StatusCode::REQUEST_TIMEOUT,
                    "request body timed out",
                    "invalid_request_error",
                )
            }
        };
        let body = match WireRequest::parse(&bytes, self.config.max_body_bytes) {
            Ok(body) => body,
            Err(error) => {
                let status = if matches!(error, crate::chat::wire::Error::Unsupported(_)) {
                    StatusCode::BAD_REQUEST
                } else {
                    StatusCode::UNPROCESSABLE_ENTITY
                };
                return error_response(status, &error.to_string(), "invalid_request_error");
            }
        };
        if body.model() != self.config.model {
            return error_response(
                StatusCode::NOT_FOUND,
                "requested model is not loaded",
                "invalid_request_error",
            );
        }
        if let Err(error) = self.client.check().await {
            return error_response(StatusCode::SERVICE_UNAVAILABLE, &error, "server_error");
        }
        let created = match SystemTime::now().duration_since(UNIX_EPOCH) {
            Ok(time) => time.as_secs(),
            Err(error) => {
                return error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &error.to_string(),
                    "server_error",
                )
            }
        };
        let now = match i64::try_from(created) {
            Ok(now) => now,
            Err(_) => {
                return error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "clock exceeds template domain",
                    "server_error",
                )
            }
        };
        let limits = ModelLimits {
            model: &self.config.model,
            context_tokens: self.config.context_tokens,
            vocabulary: self.config.vocabulary,
            output_capacity: self.config.output_capacity,
            forced_quantum: self.config.forced_quantum,
        };
        let selection = TemplateSelection {
            variant: self.config.template_variant.as_deref(),
            source_override: self.config.template_override.as_ref(),
        };
        let prepared =
            match body.prepare(&self.templates, &self.tokenizer, &selection, now, &limits) {
                Ok(prepared) => prepared,
                Err(error) => {
                    return error_response(StatusCode::BAD_REQUEST, &error, "invalid_request_error")
                }
            };
        let next = self.next.get();
        let Some(following) = next.checked_add(1) else {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "response identity exhausted",
                "server_error",
            );
        };
        self.next.set(following);
        let id = format!("chatcmpl-{}-{next}", self.identity);
        if body.stream() {
            let (sender, receiver) = mpsc::channel(1);
            let host = self.clone();
            let future = async move {
                let mut sent = 0usize;
                let result: Result<(), String> = async {
                    let mut response = SseResponse::new(
                        id,
                        host.config.model.clone(),
                        created,
                        body.include_usage(),
                        host.config.max_response_bytes,
                    )?;
                    let mut session = Session::open(
                        &host.client,
                        &prepared.chat,
                        &host.tokenizer,
                        prepared.options,
                        body.stops().to_vec(),
                        host.config.max_response_bytes,
                    )
                    .await?;
                    while let Some(frames) = session.next_sse(&mut response).await? {
                        for frame in frames {
                            send_frame(&sender, frame, &mut sent, host.config.max_response_bytes)
                                .await?;
                        }
                    }
                    Ok(())
                }
                .await;
                if let Err(error) = result {
                    let frame = format!(
                        "data: {}\n\n",
                        json!({"error":{"message":error,"type":"server_error"}})
                    )
                    .into_bytes();
                    if send_frame(&sender, frame, &mut sent, host.config.max_response_bytes)
                        .await
                        .is_ok()
                    {
                        let _ = send_frame(
                            &sender,
                            b"data: [DONE]\n\n".to_vec(),
                            &mut sent,
                            host.config.max_response_bytes,
                        )
                        .await;
                    }
                }
            };
            Response::builder()
                .status(StatusCode::OK)
                .header("Content-Type", "text/event-stream")
                .header("Cache-Control", "no-cache")
                .header("X-Accel-Buffering", "no")
                .body(ResponseBody::Stream {
                    future: Some(Box::pin(future)),
                    receiver,
                })
                .unwrap()
        } else {
            let result: Result<Vec<u8>, String> = async {
                let mut session = Session::open(
                    &self.client,
                    &prepared.chat,
                    &self.tokenizer,
                    prepared.options,
                    body.stops().to_vec(),
                    self.config.max_response_bytes,
                )
                .await?;
                let response = CompleteResponse::new(
                    id,
                    self.config.model.clone(),
                    created,
                    self.config.max_response_bytes,
                )?;
                session.complete(response).await
            }
            .await;
            match result {
                Ok(bytes) => full_response(StatusCode::OK, bytes),
                Err(error) => {
                    error_response(StatusCode::INTERNAL_SERVER_ERROR, &error, "server_error")
                }
            }
        }
    }
}
async fn read_body(mut body: Incoming, limit: usize) -> Result<Vec<u8>, (StatusCode, String)> {
    let mut bytes = Vec::new();
    while let Some(frame) = poll_fn(|cx| Pin::new(&mut body).poll_frame(cx)).await {
        let frame = frame.map_err(|error| (StatusCode::BAD_REQUEST, error.to_string()))?;
        if let Ok(data) = frame.into_data() {
            if data.len() > limit - bytes.len() {
                return Err((
                    StatusCode::PAYLOAD_TOO_LARGE,
                    "request body exceeds byte limit".into(),
                ));
            }
            bytes.extend_from_slice(&data);
        }
    }
    Ok(bytes)
}
async fn send_frame(
    sender: &mpsc::Sender<Bytes>,
    bytes: Vec<u8>,
    sent: &mut usize,
    limit: usize,
) -> Result<(), String> {
    *sent = sent
        .checked_add(bytes.len())
        .filter(|&n| n <= limit)
        .ok_or("response byte limit exceeded")?;
    sender
        .send(bytes.into())
        .await
        .map_err(|_| "response disconnected".into())
}
fn error_response(status: StatusCode, message: &str, kind: &str) -> Response<ResponseBody> {
    json_response(status, json!({"error":{"message":message,"type":kind}}))
}
fn json_response(status: StatusCode, value: serde_json::Value) -> Response<ResponseBody> {
    full_response(
        status,
        serde_json::to_vec(&value).expect("JSON value serializes"),
    )
}
fn full_response(status: StatusCode, bytes: Vec<u8>) -> Response<ResponseBody> {
    Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .body(ResponseBody::Full(Some(bytes.into())))
        .unwrap()
}
enum ResponseBody {
    Full(Option<Bytes>),
    Stream {
        future: Option<Pin<Box<dyn Future<Output = ()>>>>,
        receiver: mpsc::Receiver<Bytes>,
    },
}
impl Body for ResponseBody {
    type Data = Bytes;
    type Error = Infallible;
    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, Infallible>>> {
        match self.get_mut() {
            Self::Full(bytes) => Poll::Ready(bytes.take().map(|bytes| Ok(Frame::data(bytes)))),
            Self::Stream { future, receiver } => {
                if let Some(run) = future.as_mut() {
                    if run.as_mut().poll(cx).is_ready() {
                        *future = None;
                    }
                }
                receiver
                    .poll_recv(cx)
                    .map(|bytes| bytes.map(|bytes| Ok(Frame::data(bytes))))
            }
        }
    }
    fn is_end_stream(&self) -> bool {
        match self {
            Self::Full(bytes) => bytes.is_none(),
            Self::Stream { future, receiver } => future.is_none() && receiver.is_empty(),
        }
    }
    fn size_hint(&self) -> SizeHint {
        match self {
            Self::Full(bytes) => {
                SizeHint::with_exact(bytes.as_ref().map_or(0, |bytes| bytes.len()) as u64)
            }
            _ => SizeHint::default(),
        }
    }
}
