//! Host-local semantic stream over one scheduled request.

use super::{
    wire::PreparedGeneration, ByteBpeTokenizer, ChatPublication, CompleteResponse, SseResponse,
    TokenChatStream, Vocabulary,
};
use crate::service::{EngineClient, EngineRequest};
use magnitude_generation::FinishReason;
use magnitude_generation::GenerationSeed;
use magnitude_model_contracts::PreparedModelInput;

pub struct Session<'a> {
    request: Option<EngineRequest>,
    parser: TokenChatStream<'a>,
    terminal: bool,
}

impl<'a> Session<'a> {
    pub async fn open(
        client: &EngineClient,
        prepared: &PreparedGeneration,
        input: PreparedModelInput,
        tokenizer: &'a ByteBpeTokenizer,
        vocabulary: &mut Vocabulary,
        stops: Vec<String>,
        max_output_bytes: usize,
    ) -> Result<Self, String> {
        let seed = vocabulary.prepare_generation_for_input(
            prepared.chat.input(),
            prepared.options.clone(),
            &input,
        )?;
        Self::open_seed(
            client,
            prepared,
            input,
            tokenizer,
            seed,
            stops,
            max_output_bytes,
        )
        .await
    }

    /// Open from device-free generation intent bound by the host vocabulary. This keeps
    /// mutable grammar-cache access outside the asynchronous admission wait.
    pub async fn open_seed(
        client: &EngineClient,
        prepared: &PreparedGeneration,
        input: PreparedModelInput,
        tokenizer: &'a ByteBpeTokenizer,
        seed: GenerationSeed,
        stops: Vec<String>,
        max_output_bytes: usize,
    ) -> Result<Self, String> {
        let parser = TokenChatStream::new(&prepared.chat, tokenizer, stops, max_output_bytes)?;
        let capacity = prepared.options.output_capacity;
        let request = client.admit(seed, input, capacity).await?;
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
            let mut publication = match self.request.as_mut().expect("live session").receive().await
            {
                Ok(publication) => publication,
                Err(error) => {
                    self.terminal = true;
                    return Ok(ChatPublication {
                        events: self.parser.finish(FinishReason::Failed)?,
                        error: Some(error),
                        usage: None,
                        method: None,
                        timings: None,
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
                    usage: self.terminal.then_some(publication.usage).flatten(),
                    method: self.terminal.then_some(publication.method).flatten(),
                    timings: self.terminal.then_some(publication.timings).flatten(),
                    error: self.terminal.then_some(publication.error).flatten(),
                });
            }
        }
    }
}
