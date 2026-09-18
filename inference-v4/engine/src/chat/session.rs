//! Host-local semantic stream over a scheduled request. Native parser handles
//! remain in this context; only prepared input and token publications cross it.
use super::{CompleteResponse, Event, PreparedChat, SseResponse, TokenChatStream, Usage};
use crate::{
    generation::{FinishReason, Options},
    inputs::ByteBpeTokenizer,
    service::{
        owner::Executor,
        runtime::{Client, Request},
    },
};

pub struct ChatPublication {
    /// Present on terminal publication after the execution owner acknowledges stop.
    pub usage: Option<Usage>,
    pub events: Vec<Event>,
    /// Accepted content may accompany a terminal execution failure.
    pub error: Option<String>,
}
pub struct Session<'a, E: Executor + 'static> {
    request: Option<Request<E>>,
    parser: TokenChatStream<'a>,
    terminal: bool,
}
impl<'a, E: Executor + 'static> Session<'a, E> {
    pub async fn open(
        client: &Client<E>,
        prepared: &PreparedChat,
        tokenizer: &'a ByteBpeTokenizer,
        options: Options,
        stops: Vec<String>,
        max_output_bytes: usize,
    ) -> Result<Self, String> {
        let parser = TokenChatStream::new(prepared, tokenizer, stops, max_output_bytes)?;
        let request = client.admit(prepared.input().clone(), options).await?;
        Ok(Self {
            request: Some(request),
            parser,
            terminal: false,
        })
    }
    pub async fn next(&mut self) -> Result<Option<ChatPublication>, String> {
        if self.terminal {
            return Ok(None);
        }
        let result = self.next_impl().await;
        if result.is_err() || self.terminal {
            self.terminal = true;
            self.request.take();
        }
        result.map(Some)
    }
    /// Collect a nonstream response without changing generation or parsing.
    pub async fn complete(&mut self, mut response: CompleteResponse) -> Result<Vec<u8>, String> {
        let result = async {
            while let Some(publication) = self.next().await? {
                if let Some(bytes) = response.feed(&publication)? {
                    return Ok(bytes);
                }
            }
            Err("session ended without a terminal response".into())
        }
        .await;
        if result.is_err() {
            self.terminal = true;
            self.request.take();
        }
        result
    }
    /// Frame the same semantic session for streaming transport. Framing failure
    /// releases its request; native parser and service state never cross threads.
    pub async fn next_sse(
        &mut self,
        response: &mut SseResponse,
    ) -> Result<Option<Vec<Vec<u8>>>, String> {
        let Some(publication) = self.next().await? else {
            return Ok(None);
        };
        match response.feed(&publication) {
            Ok(frames) => Ok(Some(frames)),
            Err(error) => {
                self.terminal = true;
                self.request.take();
                Err(error)
            }
        }
    }
    async fn next_impl(&mut self) -> Result<ChatPublication, String> {
        loop {
            let mut publication = match self
                .request
                .as_mut()
                .expect("live session")
                .receive(32)
                .await
            {
                Ok(publication) => publication,
                Err(error) => {
                    self.terminal = true;
                    return Ok(ChatPublication {
                        events: self.parser.finish(FinishReason::Failed)?,
                        error: Some(error),
                        usage: None,
                    });
                }
            };
            let mut events = Vec::new();
            for token in &publication.tokens {
                events.extend(self.parser.feed(token)?);
                if self.parser.stopped() {
                    self.terminal = true;
                    publication = self.request.as_mut().expect("live session").stop().await?;
                    break;
                }
            }
            if !self.terminal {
                if let Some(reason) = publication.finish {
                    events.extend(self.parser.finish(reason)?);
                    self.terminal = true;
                }
            }
            if !events.is_empty() || self.terminal {
                return Ok(ChatPublication {
                    events,
                    usage: if self.terminal {
                        publication.usage
                    } else {
                        None
                    },
                    error: if self.terminal {
                        publication.error
                    } else {
                        None
                    },
                });
            }
        }
    }
}
