//! Structured runtime failures preserve resource facts through preparation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    Capacity { required: usize, available: usize },
    Failure(String),
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Capacity {
                required,
                available,
            } => write!(
                f,
                "allocation requires {required} bytes; {available} charged bytes available"
            ),
            Self::Failure(message) => f.write_str(message),
        }
    }
}
impl std::error::Error for Error {}
impl From<String> for Error {
    fn from(message: String) -> Self {
        Self::Failure(message)
    }
}
impl From<&str> for Error {
    fn from(message: &str) -> Self {
        Self::Failure(message.into())
    }
}
impl From<Error> for String {
    fn from(error: Error) -> Self {
        error.to_string()
    }
}
