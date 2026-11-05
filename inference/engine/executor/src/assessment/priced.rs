//! What a decision takes of one input it prices with: measured, pending (the
//! measurement job has not stored its points yet: the decision runs its
//! policy that needs no cost until `Reprice` brings them), or unavailable,
//! with why. Each decision takes the states of exactly its inputs.

/// One input of a decision, in its state.
#[derive(Clone, Debug, PartialEq)]
pub enum Priced<T> {
    Measured(T),
    /// What is awaited.
    Pending(String),
    /// Why there is none.
    Unavailable(String),
}

impl<T> Priced<T> {
    /// The input, where measured.
    pub fn measured(&self) -> Option<&T> {
        match self {
            Self::Measured(value) => Some(value),
            Self::Pending(_) | Self::Unavailable(_) => None,
        }
    }

    /// What is awaited or why there is none; none where measured.
    pub fn why(&self) -> Option<&str> {
        match self {
            Self::Measured(_) => None,
            Self::Pending(why) | Self::Unavailable(why) => Some(why),
        }
    }

    pub fn as_ref(&self) -> Priced<&T> {
        match self {
            Self::Measured(value) => Priced::Measured(value),
            Self::Pending(why) => Priced::Pending(why.clone()),
            Self::Unavailable(why) => Priced::Unavailable(why.clone()),
        }
    }

    /// The input `then` derives from this one, in this one's state where it
    /// is not measured.
    pub fn and_then<U>(self, then: impl FnOnce(T) -> Priced<U>) -> Priced<U> {
        match self {
            Self::Measured(value) => then(value),
            Self::Pending(why) => Priced::Pending(why),
            Self::Unavailable(why) => Priced::Unavailable(why),
        }
    }

    /// The input `derive` makes of this one, unavailable where it fails.
    pub fn derive<U, E: std::fmt::Display>(self, derive: impl FnOnce(T) -> Result<U, E>) -> Priced<U> {
        self.and_then(|value| match derive(value) {
            Ok(derived) => Priced::Measured(derived),
            Err(error) => Priced::Unavailable(error.to_string()),
        })
    }
}
