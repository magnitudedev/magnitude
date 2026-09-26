//! Message transports between a host and its numerical worker.
//!
//! A transport is split into a receiving and a sending half, each owned by one
//! thread. The in-process transport moves typed messages through channels; the
//! framed transport encodes each message (postcard) behind a little-endian
//! `u32` length over any byte stream, such as a worker process's stdio.

use super::protocol::{HostMessage, WorkerMessage};
use serde::{de::DeserializeOwned, Serialize};
use std::{
    fmt,
    io::{self, Read, Write},
    marker::PhantomData,
    sync::mpsc,
};

/// The largest frame either side accepts: a request with its maximum media
/// payload fits well within it.
pub const MAX_FRAME_BYTES: u32 = 1 << 30;

#[derive(Debug)]
pub enum TransportError {
    Io(io::Error),
    Encode(postcard::Error),
    Decode(postcard::Error),
    FrameTooLarge(u64),
    /// The other end is gone.
    Closed,
}

impl fmt::Display for TransportError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "transport i/o failed: {error}"),
            Self::Encode(error) => write!(formatter, "message encoding failed: {error}"),
            Self::Decode(error) => write!(formatter, "message decoding failed: {error}"),
            Self::FrameTooLarge(bytes) => {
                write!(formatter, "frame of {bytes} bytes exceeds {MAX_FRAME_BYTES}")
            }
            Self::Closed => formatter.write_str("the other end of the transport is closed"),
        }
    }
}

impl std::error::Error for TransportError {}

pub trait MessageReceiver<M>: Send + 'static {
    /// The next message; `None` at a clean end of stream.
    fn receive(&mut self) -> Result<Option<M>, TransportError>;
}

pub trait MessageSender<M>: Send + 'static {
    fn send(&mut self, message: M) -> Result<(), TransportError>;
}

/// One end of a transport that receives `In` and sends `Out`.
pub trait Transport<In, Out>: Send + 'static {
    type Receiver: MessageReceiver<In>;
    type Sender: MessageSender<Out>;
    fn split(self) -> (Self::Receiver, Self::Sender);
}

/// The worker's end: receives host messages, sends worker messages.
pub trait WorkerTransport: Transport<HostMessage, WorkerMessage> {}
impl<T: Transport<HostMessage, WorkerMessage>> WorkerTransport for T {}

/// The host's end: receives worker messages, sends host messages.
pub trait HostTransport: Transport<WorkerMessage, HostMessage> {}
impl<T: Transport<WorkerMessage, HostMessage>> HostTransport for T {}

/// An in-process transport end over channels. No encoding: values move.
pub struct ChannelTransport<In, Out> {
    receiver: mpsc::Receiver<In>,
    sender: mpsc::Sender<Out>,
}

/// A connected pair: the host's end and the worker's end.
pub fn channel_pair() -> (
    ChannelTransport<WorkerMessage, HostMessage>,
    ChannelTransport<HostMessage, WorkerMessage>,
) {
    let (to_worker, from_host) = mpsc::channel();
    let (to_host, from_worker) = mpsc::channel();
    (
        ChannelTransport {
            receiver: from_worker,
            sender: to_worker,
        },
        ChannelTransport {
            receiver: from_host,
            sender: to_host,
        },
    )
}

pub struct ChannelReceiver<M>(mpsc::Receiver<M>);
pub struct ChannelSender<M>(mpsc::Sender<M>);

impl<M: Send + 'static> MessageReceiver<M> for ChannelReceiver<M> {
    fn receive(&mut self) -> Result<Option<M>, TransportError> {
        Ok(self.0.recv().ok())
    }
}

impl<M: Send + 'static> MessageSender<M> for ChannelSender<M> {
    fn send(&mut self, message: M) -> Result<(), TransportError> {
        self.0.send(message).map_err(|_| TransportError::Closed)
    }
}

impl<In: Send + 'static, Out: Send + 'static> Transport<In, Out> for ChannelTransport<In, Out> {
    type Receiver = ChannelReceiver<In>;
    type Sender = ChannelSender<Out>;
    fn split(self) -> (Self::Receiver, Self::Sender) {
        (ChannelReceiver(self.receiver), ChannelSender(self.sender))
    }
}

/// A transport end over a byte stream pair with length-prefixed frames.
pub struct FramedTransport<R, W, In, Out> {
    reader: R,
    writer: W,
    messages: PhantomData<fn(In) -> Out>,
}

impl<R, W, In, Out> FramedTransport<R, W, In, Out> {
    pub fn new(reader: R, writer: W) -> Self {
        Self {
            reader,
            writer,
            messages: PhantomData,
        }
    }
}

pub struct FrameReader<R, M> {
    reader: R,
    buffer: Vec<u8>,
    message: PhantomData<fn() -> M>,
}

pub struct FrameWriter<W, M> {
    writer: W,
    buffer: Vec<u8>,
    message: PhantomData<fn(M)>,
}

impl<R: Read + Send + 'static, M: DeserializeOwned + 'static> MessageReceiver<M>
    for FrameReader<R, M>
{
    fn receive(&mut self) -> Result<Option<M>, TransportError> {
        let mut length = [0u8; 4];
        let mut read = 0;
        while read < length.len() {
            match self.reader.read(&mut length[read..]) {
                Ok(0) if read == 0 => return Ok(None),
                Ok(0) => {
                    return Err(TransportError::Io(io::ErrorKind::UnexpectedEof.into()))
                }
                Ok(count) => read += count,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(error) => return Err(TransportError::Io(error)),
            }
        }
        let length = u32::from_le_bytes(length);
        if length > MAX_FRAME_BYTES {
            return Err(TransportError::FrameTooLarge(u64::from(length)));
        }
        self.buffer.resize(length as usize, 0);
        self.reader
            .read_exact(&mut self.buffer)
            .map_err(TransportError::Io)?;
        postcard::from_bytes(&self.buffer)
            .map(Some)
            .map_err(TransportError::Decode)
    }
}

impl<W: Write + Send + 'static, M: Serialize + 'static> MessageSender<M> for FrameWriter<W, M> {
    fn send(&mut self, message: M) -> Result<(), TransportError> {
        self.buffer.clear();
        self.buffer.extend_from_slice(&[0; 4]);
        let buffer = std::mem::take(&mut self.buffer);
        let mut buffer =
            postcard::to_extend(&message, buffer).map_err(TransportError::Encode)?;
        let length = buffer.len() - 4;
        let length = u32::try_from(length)
            .ok()
            .filter(|length| *length <= MAX_FRAME_BYTES)
            .ok_or(TransportError::FrameTooLarge(length as u64))?;
        buffer[..4].copy_from_slice(&length.to_le_bytes());
        let written = self
            .writer
            .write_all(&buffer)
            .and_then(|()| self.writer.flush())
            .map_err(|error| match error.kind() {
                io::ErrorKind::BrokenPipe => TransportError::Closed,
                _ => TransportError::Io(error),
            });
        self.buffer = buffer;
        written
    }
}

impl<R, W, In, Out> Transport<In, Out> for FramedTransport<R, W, In, Out>
where
    R: Read + Send + 'static,
    W: Write + Send + 'static,
    In: DeserializeOwned + 'static,
    Out: Serialize + 'static,
{
    type Receiver = FrameReader<R, In>;
    type Sender = FrameWriter<W, Out>;
    fn split(self) -> (Self::Receiver, Self::Sender) {
        (
            FrameReader {
                reader: self.reader,
                buffer: Vec::new(),
                message: PhantomData,
            },
            FrameWriter {
                writer: self.writer,
                buffer: Vec::new(),
                message: PhantomData,
            },
        )
    }
}

/// A worker process's end over its own stdin and stdout. Diagnostics go to
/// stderr; nothing else may write to stdout.
pub fn stdio_worker_transport(
) -> FramedTransport<io::Stdin, io::Stdout, HostMessage, WorkerMessage> {
    FramedTransport::new(io::stdin(), io::stdout())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worker::protocol::{EngineBuild, HostRequestId};

    #[test]
    fn frames_round_trip_over_a_byte_stream() {
        let bytes = {
            let (_, mut sender) =
                FramedTransport::<io::Empty, Vec<u8>, WorkerMessage, HostMessage>::new(
                    io::empty(),
                    Vec::new(),
                )
                .split();
            sender
                .send(HostMessage::Hello {
                    build: EngineBuild::current(),
                })
                .unwrap();
            sender
                .send(HostMessage::Credit {
                    request_id: HostRequestId(7),
                    batches: 3,
                })
                .unwrap();
            sender.writer
        };
        let (mut receiver, _) =
            FramedTransport::<_, io::Sink, HostMessage, WorkerMessage>::new(
                io::Cursor::new(bytes),
                io::sink(),
            )
            .split();
        assert!(matches!(
            receiver.receive().unwrap(),
            Some(HostMessage::Hello { build }) if build == EngineBuild::current()
        ));
        assert!(matches!(
            receiver.receive().unwrap(),
            Some(HostMessage::Credit { request_id: HostRequestId(7), batches: 3 })
        ));
        assert!(receiver.receive().unwrap().is_none());
    }
}
