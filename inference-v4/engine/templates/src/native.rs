//! The only unsafe boundary. Handles never escape and output is copied while its
//! native owner is alive. Live owners stay in their creating execution context.
#![deny(unsafe_op_in_unsafe_fn)]
use crate::*;
use sha2::{Digest, Sha256};
use std::{marker::PhantomData, rc::Rc, sync::OnceLock};

const UPSTREAM: &str = "930e2fa5995789efbf249a8bf61325bb626e417b";

#[repr(C)]
#[derive(Default)]
struct Buffer {
    data: *const u8,
    size: u64,
    owner: u64,
}
#[repr(C)]
struct RawEvent {
    kind: u32,
    index: u32,
    text: *const u8,
    text_size: u64,
    id: *const u8,
    id_size: u64,
}
#[repr(C)]
#[derive(Default)]
struct Events {
    data: *const RawEvent,
    size: u64,
}
extern "C" {
    fn templates_abi_version() -> u32;
    fn templates_build_info(output: *mut Buffer, error: *mut Buffer) -> i32;
    fn templates_buffer_release(owner: u64) -> i32;
    fn templates_template_create(
        json: *const u8,
        size: u64,
        output: *mut u64,
        error: *mut Buffer,
    ) -> i32;
    fn templates_template_release(handle: u64, error: *mut Buffer) -> i32;
    fn templates_template_inspect(handle: u64, output: *mut Buffer, error: *mut Buffer) -> i32;
    fn templates_template_render(
        handle: u64,
        json: *const u8,
        size: u64,
        output: *mut Buffer,
        error: *mut Buffer,
    ) -> i32;
    fn templates_request_create(
        handle: u64,
        json: *const u8,
        size: u64,
        output: *mut u64,
        error: *mut Buffer,
    ) -> i32;
    fn templates_request_describe(handle: u64, output: *mut Buffer, error: *mut Buffer) -> i32;
    fn templates_request_release(handle: u64, error: *mut Buffer) -> i32;
    fn templates_stream_create(
        handle: u64,
        limit: u64,
        output: *mut u64,
        error: *mut Buffer,
    ) -> i32;
    fn templates_stream_feed(
        handle: u64,
        bytes: *const u8,
        size: u64,
        output: *mut Events,
        error: *mut Buffer,
    ) -> i32;
    fn templates_stream_finish(
        handle: u64,
        cause: u32,
        output: *mut Events,
        error: *mut Buffer,
    ) -> i32;
    fn templates_stream_release(handle: u64, error: *mut Buffer) -> i32;
}

// The ABI guarantees that successful calls return valid spans, retained until
// owner release (buffers) or the next mutation of the same stream (events).
unsafe fn span<'a, T>(pointer: *const T, len: u64) -> Result<&'a [T], Error> {
    if len == 0 {
        return Ok(&[]);
    }
    let len = usize::try_from(len)
        .map_err(|_| Error::incompatible("native span exceeds address space"))?;
    if pointer.is_null() || len > isize::MAX as usize / std::mem::size_of::<T>() {
        return Err(Error::incompatible("invalid native span"));
    }
    // SAFETY: caller holds the native owner and excludes invalidating calls.
    Ok(unsafe { std::slice::from_raw_parts(pointer, len) })
}
impl Drop for Buffer {
    fn drop(&mut self) {
        if self.owner != 0 {
            // SAFETY: this buffer uniquely owns the returned release ID.
            unsafe {
                templates_buffer_release(self.owner);
            }
        }
    }
}
impl Buffer {
    fn bytes(&self) -> Result<&[u8], Error> {
        // SAFETY: all buffers are initialized by the ABI, and self retains owner.
        unsafe { span(self.data, self.size) }
    }
}
fn check(status: i32, error: &Buffer) -> Result<(), Error> {
    if status == 0 {
        return Ok(());
    }
    let kind = match status {
        1 => ErrorKind::InvalidArgument,
        2 => ErrorKind::InvalidHandle,
        3 => ErrorKind::OutOfMemory,
        4 => ErrorKind::Native,
        _ => ErrorKind::Incompatible,
    };
    Err(Error {
        kind,
        message: String::from_utf8_lossy(error.bytes()?).into_owned(),
    })
}
fn output(call: impl FnOnce(*mut Buffer, *mut Buffer) -> i32) -> Result<Vec<u8>, Error> {
    let mut output = Buffer::default();
    let mut error = Buffer::default();
    check(call(&mut output, &mut error), &error)?;
    Ok(output.bytes()?.to_vec())
}
fn decode<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T, Error> {
    serde_json::from_slice(bytes).map_err(|e| Error::incompatible(format!("native JSON: {e}")))
}
fn encode(value: &impl Serialize) -> Result<Vec<u8>, Error> {
    serde_json::to_vec(value).map_err(|e| Error::invalid(e.to_string()))
}
pub fn build_info() -> Result<&'static BuildInfo, Error> {
    static BUILD: OnceLock<Result<BuildInfo, Error>> = OnceLock::new();
    BUILD
        .get_or_init(|| {
            // SAFETY: the linked ABI accepts initialized out pointers and no handles.
            if unsafe { templates_abi_version() } != 1 {
                return Err(Error::incompatible("native template ABI mismatch"));
            }
            let info: BuildInfo = decode(&output(|o, e| unsafe { templates_build_info(o, e) })?)?;
            if info.version != 1
                || info.abi != 1
                || info.extraction != 1
                || info.upstream != UPSTREAM
            {
                return Err(Error::incompatible(
                    "native template build identity mismatch",
                ));
            }
            Ok(info)
        })
        .as_ref()
        .map_err(Clone::clone)
}

enum Kind {
    Template,
    Request,
    Stream,
}
struct Handle {
    id: u64,
    kind: Kind,
    // Live owners do not cross the engine's host/execution contexts.
    _context: PhantomData<Rc<()>>,
}
impl Handle {
    fn create(kind: Kind, call: impl FnOnce(*mut u64, *mut Buffer) -> i32) -> Result<Self, Error> {
        let mut owner = Self {
            id: 0,
            kind,
            _context: PhantomData,
        };
        let mut error = Buffer::default();
        check(call(&mut owner.id, &mut error), &error)?;
        if owner.id == 0 {
            return Err(Error::incompatible(
                "native creation returned an empty handle",
            ));
        }
        Ok(owner)
    }
}
impl Drop for Handle {
    fn drop(&mut self) {
        if self.id == 0 {
            return;
        }
        let mut error = Buffer::default();
        // SAFETY: handle kinds match their creators; unique owners release once.
        // Drop never panics, including while unwinding a failed description read.
        unsafe {
            match self.kind {
                Kind::Template => templates_template_release(self.id, &mut error),
                Kind::Request => templates_request_release(self.id, &mut error),
                Kind::Stream => templates_stream_release(self.id, &mut error),
            };
        }
    }
}

pub struct Template {
    handle: Handle,
    identity: String,
}
impl Template {
    pub fn new(source: &str, special_tokens: &SpecialTokens) -> Result<Self, Error> {
        let build = build_info()?;
        let payload = encode(
            &serde_json::json!({"version":1,"source":source,"special_tokens":special_tokens}),
        )?;
        let identity = Sha256::digest(encode(&serde_json::json!({
            "source":source,"special_tokens":special_tokens,"native":build
        }))?)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
        // SAFETY: borrowed payload lives through the call, output is initialized.
        let handle = Handle::create(Kind::Template, |o, e| unsafe {
            templates_template_create(payload.as_ptr(), payload.len() as u64, o, e)
        })?;
        Ok(Self { handle, identity })
    }
    pub fn identity(&self) -> &str {
        &self.identity
    }
    pub fn capabilities(&self) -> Result<BTreeMap<String, bool>, Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Capabilities {
            version: u32,
            capabilities: BTreeMap<String, bool>,
        }
        let result: Capabilities = decode(&output(|o, e| unsafe {
            templates_template_inspect(self.handle.id, o, e)
        })?)?;
        if result.version != 1 {
            return Err(Error::incompatible("capability payload version"));
        }
        Ok(result.capabilities)
    }
    pub fn render(&self, context: &Map<String, Value>, now: i64) -> Result<String, Error> {
        let payload = encode(&serde_json::json!({"version":1,"context":context,"now":now}))?;
        let bytes = output(|o, e| unsafe {
            templates_template_render(self.handle.id, payload.as_ptr(), payload.len() as u64, o, e)
        })?;
        String::from_utf8(bytes).map_err(|_| Error::incompatible("native prompt is not UTF-8"))
    }
    pub fn prepare(&self, request: &Request) -> Result<PreparedRequest, Error> {
        #[derive(Serialize)]
        struct Payload<'a> {
            version: u32,
            #[serde(flatten)]
            request: &'a Request,
        }
        let payload = encode(&Payload {
            version: 1,
            request,
        })?;
        let handle = Handle::create(Kind::Request, |o, e| unsafe {
            templates_request_create(self.handle.id, payload.as_ptr(), payload.len() as u64, o, e)
        })?;
        let description: PreparedDescription = decode(&output(|o, e| unsafe {
            templates_request_describe(handle.id, o, e)
        })?)?;
        if description.version != 1
            || description.grammar_dialect != "gbnf"
            || description.grammar_lazy
            || !description.grammar_triggers.is_empty()
        {
            return Err(Error::incompatible("unsupported native grammar contract"));
        }
        Ok(PreparedRequest {
            handle,
            description,
            template_identity: self.identity.clone(),
        })
    }
}

/// Retains its native plan independently of the source template.
pub struct PreparedRequest {
    handle: Handle,
    description: PreparedDescription,
    template_identity: String,
}
impl PreparedRequest {
    pub fn template_identity(&self) -> &str {
        &self.template_identity
    }
    pub fn description(&self) -> &PreparedDescription {
        &self.description
    }
    pub fn stream(&self, max_output_bytes: usize) -> Result<OutputStream, Error> {
        if !(1..=64 * 1024 * 1024).contains(&max_output_bytes) {
            return Err(Error::invalid(
                "max_output_bytes must be between 1 byte and 64 MiB",
            ));
        }
        let handle = Handle::create(Kind::Stream, |o, e| unsafe {
            templates_stream_create(self.handle.id, max_output_bytes as u64, o, e)
        })?;
        Ok(OutputStream { handle })
    }
}

/// Retains its plan independently of the request. Mutable access excludes calls
/// that would invalidate borrowed native events while they are being copied.
pub struct OutputStream {
    handle: Handle,
}
impl OutputStream {
    pub fn feed(&mut self, bytes: &[u8]) -> Result<Vec<Event>, Error> {
        self.events(|id, o, e| unsafe {
            templates_stream_feed(id, bytes.as_ptr(), bytes.len() as u64, o, e)
        })
    }
    pub fn finish(&mut self, cause: TerminalCause) -> Result<Vec<Event>, Error> {
        self.events(|id, o, e| unsafe { templates_stream_finish(id, cause as u32, o, e) })
    }
    fn events(
        &mut self,
        call: impl FnOnce(u64, *mut Events, *mut Buffer) -> i32,
    ) -> Result<Vec<Event>, Error> {
        let mut events = Events::default();
        let mut error = Buffer::default();
        check(call(self.handle.id, &mut events, &mut error), &error)?;
        // SAFETY: self owns the stream exclusively, and no mutation occurs while
        // copying its event spans. Returned Rust events never borrow native memory.
        unsafe { span(events.data, events.size)? }
            .iter()
            .map(|event| {
                let text = |p, n| -> Result<String, Error> {
                    let bytes = unsafe { span(p, n)? };
                    std::str::from_utf8(bytes)
                        .map(str::to_owned)
                        .map_err(|_| Error::incompatible("native event is not UTF-8"))
                };
                Ok(match event.kind {
                    1 => Event::Content {
                        text: text(event.text, event.text_size)?,
                    },
                    2 => Event::Reasoning {
                        text: text(event.text, event.text_size)?,
                    },
                    3 => Event::ToolStart {
                        index: event.index,
                        name: text(event.text, event.text_size)?,
                        id: text(event.id, event.id_size)?,
                    },
                    4 => Event::ToolArguments {
                        index: event.index,
                        text: text(event.text, event.text_size)?,
                    },
                    5 => Event::ToolComplete { index: event.index },
                    6 => Event::Finish {
                        cause: match event.index {
                            0 => TerminalCause::Natural,
                            1 => TerminalCause::Length,
                            2 => TerminalCause::UserStop,
                            3 => TerminalCause::Cancelled,
                            4 => TerminalCause::Failed,
                            _ => return Err(Error::incompatible("unknown native terminal cause")),
                        },
                    },
                    _ => return Err(Error::incompatible("unknown native event kind")),
                })
            })
            .collect()
    }
}
