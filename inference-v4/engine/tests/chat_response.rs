use seismic_engine::chat::{ChatPublication, Event, SseResponse, TerminalCause, Usage};
use serde_json::{json, Value};
fn response(usage: bool) -> SseResponse {
    SseResponse::new("chatcmpl-fixture".into(), "model".into(), 123, usage, 8192).unwrap()
}
fn decode(frame: &[u8]) -> Value {
    let text = std::str::from_utf8(frame).unwrap();
    assert_eq!(text.matches("\n\n").count(), 1);
    serde_json::from_str(
        text.strip_prefix("data: ")
            .unwrap()
            .strip_suffix("\n\n")
            .unwrap(),
    )
    .unwrap()
}
#[test]
fn tool_reasoning_and_content_frames_preserve_json_boundaries_and_usage() {
    let publication = ChatPublication {
        usage: Some(Usage {
            prompt_tokens: 5,
            completion_tokens: 3,
        }),
        events: vec![
            Event::Reasoning {
                text: "think\n\ndata: injected 世界".into(),
            },
            Event::Content {
                text: "answer".into(),
            },
            Event::ToolStart {
                index: 0,
                name: "search".into(),
                id: "call_1".into(),
            },
            Event::ToolArguments {
                index: 0,
                text: "{\"q\":\"世界\"}".into(),
            },
            Event::ToolComplete { index: 0 },
            Event::Finish {
                cause: TerminalCause::Natural,
            },
        ],
        error: None,
    };
    let mut response = response(true);
    let frames = response.feed(&publication).unwrap();
    assert_eq!(frames.len(), 8);
    assert_eq!(frames.last().unwrap(), b"data: [DONE]\n\n");
    let values: Vec<_> = frames[..7].iter().map(|f| decode(f)).collect();
    assert_eq!(values[0]["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(
        values[1]["choices"][0]["delta"]["reasoning_content"],
        "think\n\ndata: injected 世界"
    );
    assert_eq!(
        values[3]["choices"][0]["delta"]["tool_calls"][0]["id"],
        "call_1"
    );
    assert_eq!(values[5]["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(values[6]["choices"], json!([]));
    assert_eq!(values[6]["usage"]["total_tokens"], 8);
    assert!(response.feed(&publication).is_err());
}
#[test]
fn truncation_and_failure_do_not_report_completed_tool_calls() {
    for (cause, error, reason) in [
        (TerminalCause::Length, None, Some("length")),
        (TerminalCause::UserStop, None, Some("stop")),
        (TerminalCause::Failed, Some("device failed".into()), None),
    ] {
        let publication = ChatPublication {
            usage: None,
            events: vec![
                Event::ToolStart {
                    index: 0,
                    name: "search".into(),
                    id: "call_1".into(),
                },
                Event::ToolArguments {
                    index: 0,
                    text: "{\"q\":".into(),
                },
                Event::Finish { cause },
            ],
            error,
        };
        let frames = response(false).feed(&publication).unwrap();
        let terminal = decode(&frames[frames.len() - 2]);
        if let Some(reason) = reason {
            assert_eq!(terminal["choices"][0]["finish_reason"], reason);
        } else {
            assert_eq!(terminal["error"]["message"], "device failed");
        }
    }
}
#[test]
fn limits_invalid_event_order_and_missing_usage_fail_closed() {
    let terminal = ChatPublication {
        usage: None,
        events: vec![Event::Finish {
            cause: TerminalCause::Natural,
        }],
        error: None,
    };
    assert!(response(true).feed(&terminal).is_err());
    let mut limited = SseResponse::new("id".into(), "m".into(), 0, false, 10).unwrap();
    assert!(limited.feed(&terminal).is_err());
    assert!(limited.feed(&terminal).is_err());
    let invalid = ChatPublication {
        usage: None,
        events: vec![Event::ToolArguments {
            index: 0,
            text: "x".into(),
        }],
        error: None,
    };
    assert!(response(false).feed(&invalid).is_err());
}

#[test]
fn nonstream_assembly_preserves_chunked_reasoning_tools_and_terminal_usage() {
    use seismic_engine::chat::CompleteResponse;
    let mut complete =
        CompleteResponse::new("chatcmpl-fixture".into(), "model".into(), 123, 8192).unwrap();
    let mut stream = response(true);
    let publications = [
        ChatPublication {
            usage: None,
            error: None,
            events: vec![
                Event::Reasoning {
                    text: "think 世界".into(),
                },
                Event::Content {
                    text: "answer\n".into(),
                },
                Event::ToolStart {
                    index: 0,
                    id: "call_1".into(),
                    name: "search".into(),
                },
                Event::ToolArguments {
                    index: 0,
                    text: "{\"q\":".into(),
                },
            ],
        },
        ChatPublication {
            usage: Some(Usage {
                prompt_tokens: 5,
                completion_tokens: 7,
            }),
            error: None,
            events: vec![
                Event::Reasoning {
                    text: " done".into(),
                },
                Event::ToolArguments {
                    index: 0,
                    text: "\"世界\"}".into(),
                },
                Event::ToolComplete { index: 0 },
                Event::Finish {
                    cause: TerminalCause::Natural,
                },
            ],
        },
    ];
    assert!(complete.feed(&publications[0]).unwrap().is_none());
    stream.feed(&publications[0]).unwrap();
    let bytes = complete.feed(&publications[1]).unwrap().unwrap();
    let value: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["object"], "chat.completion");
    assert_eq!(value["choices"][0]["finish_reason"], "tool_calls");
    let message = &value["choices"][0]["message"];
    assert_eq!(message["content"], "answer\n");
    assert_eq!(message["reasoning_content"], "think 世界 done");
    assert_eq!(
        message["tool_calls"][0]["function"]["arguments"],
        "{\"q\":\"世界\"}"
    );
    assert!(message["tool_calls"][0].get("index").is_none());
    let frames = stream.feed(&publications[1]).unwrap();
    let streamed = frames[..frames.len() - 1]
        .iter()
        .map(|frame| decode(frame))
        .collect::<Vec<_>>();
    assert_eq!(
        value["usage"],
        streamed
            .iter()
            .find_map(|chunk| chunk.get("usage"))
            .unwrap()
            .clone()
    );
    assert!(complete.feed(&publications[1]).is_err());
}

#[test]
fn nonstream_truncation_failure_and_bounds_never_fabricate_success() {
    use seismic_engine::chat::CompleteResponse;
    let create = |limit| {
        CompleteResponse::new("chatcmpl-fixture".into(), "model".into(), 123, limit).unwrap()
    };
    for cause in [
        TerminalCause::Natural,
        TerminalCause::Length,
        TerminalCause::UserStop,
        TerminalCause::Failed,
        TerminalCause::Cancelled,
    ] {
        let publication = ChatPublication {
            usage: Some(Usage {
                prompt_tokens: 1,
                completion_tokens: 2,
            }),
            error: None,
            events: vec![
                Event::ToolStart {
                    index: 0,
                    name: "partial".into(),
                    id: "call_1".into(),
                },
                Event::ToolArguments {
                    index: 0,
                    text: "{\"unfinished\":".into(),
                },
                Event::Finish { cause },
            ],
        };
        let result = create(8192).feed(&publication);
        if matches!(cause, TerminalCause::Length | TerminalCause::UserStop) {
            let value: Value = serde_json::from_slice(&result.unwrap().unwrap()).unwrap();
            assert_eq!(
                value["choices"][0]["finish_reason"],
                if cause == TerminalCause::Length {
                    "length"
                } else {
                    "stop"
                }
            );
            assert_eq!(value["choices"][0]["message"]["content"], Value::Null);
        } else {
            assert!(result.is_err());
        }
    }
    let missing = ChatPublication {
        usage: None,
        error: None,
        events: vec![Event::Finish {
            cause: TerminalCause::Natural,
        }],
    };
    assert!(create(8192).feed(&missing).is_err());
    let content = ChatPublication {
        usage: None,
        error: None,
        events: vec![Event::Content {
            text: "x".repeat(33),
        }],
    };
    let mut small = create(32);
    assert!(small.feed(&content).is_err());
    assert!(small.feed(&missing).is_err());
    let error = ChatPublication {
        usage: None,
        error: Some("device failed".into()),
        events: vec![
            Event::Content {
                text: "accepted".into(),
            },
            Event::Finish {
                cause: TerminalCause::Failed,
            },
        ],
    };
    assert_eq!(create(8192).feed(&error).unwrap_err(), "device failed");
    let after_terminal = ChatPublication {
        usage: Some(Usage {
            prompt_tokens: 1,
            completion_tokens: 0,
        }),
        error: None,
        events: vec![
            Event::Finish {
                cause: TerminalCause::Natural,
            },
            Event::Content {
                text: "late".into(),
            },
        ],
    };
    assert!(create(8192).feed(&after_terminal).is_err());
}
