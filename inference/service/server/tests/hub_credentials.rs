//! Runs alone in its own process: it changes the environment.

use std::io::{BufRead, BufReader, Write};
use std::sync::{Arc, Mutex};

fn serve_not_found() -> (String, Arc<Mutex<Vec<Option<String>>>>) {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let recorded = Arc::clone(&seen);
    std::thread::spawn(move || {
        for connection in listener.incoming() {
            let Ok(mut connection) = connection else {
                return;
            };
            let mut reader = BufReader::new(connection.try_clone().unwrap());
            let mut authorization = None;
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                    break;
                }
                if let Some((name, value)) = line.split_once(':')
                    && name.eq_ignore_ascii_case("authorization")
                {
                    authorization = Some(value.trim().to_owned());
                }
            }
            recorded.lock().unwrap().push(authorization);
            let _ = write!(
                connection,
                "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            );
        }
    });
    (endpoint, seen)
}

fn authorizations_of_an_anonymous_client(endpoint: &str, seen: &Mutex<Vec<Option<String>>>) -> Vec<Option<String>> {
    let client = hf_hub::HFClient::builder()
        .endpoint(endpoint.to_owned())
        .retry_max_attempts(1)
        .cache_enabled(false)
        .build()
        .unwrap();
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            let _ = client
                .model("owner".to_owned(), "repository".to_owned())
                .download_file_stream()
                .filename("model.gguf".to_owned())
                .revision("main".to_owned())
                .send()
                .await;
        });
    std::mem::take(&mut *seen.lock().unwrap())
}

#[test]
fn the_hub_client_never_presents_a_token_from_the_environment_or_the_token_file() {
    let home = tempfile::tempdir().unwrap();
    std::fs::write(home.path().join("token"), "hf_FromTheTokenFile000000000000000000").unwrap();
    let (endpoint, seen) = serve_not_found();
    // SAFETY: this test is the only one in its process and starts no thread that reads the
    // environment before these writes.
    unsafe {
        std::env::remove_var("HF_HUB_DISABLE_IMPLICIT_TOKEN");
        std::env::set_var("HF_TOKEN", "hf_FromTheEnvironment00000000000000000");
        std::env::set_var("HF_HOME", home.path());
    }
    let before = authorizations_of_an_anonymous_client(&endpoint, &seen);
    assert!(
        before
            .iter()
            .any(|authorization| authorization.as_deref() == Some("Bearer hf_FromTheEnvironment00000000000000000")),
        "the control: hf_hub resolves HF_TOKEN implicitly: {before:?}"
    );

    // SAFETY: as above.
    unsafe { magnitude_service_server::isolate_hub_credentials() };
    assert!(std::env::var_os("HF_TOKEN").is_none());
    let after = authorizations_of_an_anonymous_client(&endpoint, &seen);
    assert!(!after.is_empty(), "the client made no request");
    assert!(
        after.iter().all(Option::is_none),
        "an anonymous client presented a token: {after:?}"
    );
}
