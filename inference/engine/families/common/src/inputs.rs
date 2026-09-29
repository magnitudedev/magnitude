//! Text-only input: every row sits at its absolute position, and no media is
//! admitted.

use magnitude_artifacts::media::PreparedMedia;
use magnitude_family_contracts::{
    FamilyId, InputPreparationError, ModelDefinition, ModelInputAdapter, PreparedModelInput,
    TokenPlan,
};

/// The input adapter of a text-only definition.
#[derive(Clone, Debug)]
pub struct TextInput {
    family: FamilyId,
}

impl TextInput {
    /// The adapter for inputs to `definition`'s family.
    pub fn new(definition: &ModelDefinition) -> Self {
        Self {
            family: definition.family.clone(),
        }
    }
}

impl ModelInputAdapter for TextInput {
    fn prepare(
        &self,
        definition: &ModelDefinition,
        tokens: TokenPlan,
        media: &[PreparedMedia],
    ) -> Result<PreparedModelInput, InputPreparationError> {
        if definition.family != self.family {
            return Err(InputPreparationError::InputAlignment);
        }
        if !media.is_empty() {
            return Err(InputPreparationError::UnsupportedMedia);
        }
        if tokens
            .tokens()
            .iter()
            .any(|token| u64::from(token.0) >= definition.decoder.vocabulary)
        {
            return Err(InputPreparationError::TokenDomain);
        }
        let coordinates = (0..tokens.tokens().len())
            .map(|position| i32::try_from(position).map(|position| [position; 3]))
            .collect::<Result<_, _>>()
            .map_err(|_| InputPreparationError::InputAlignment)?;
        PreparedModelInput::from_text_coordinates(tokens, coordinates)
    }
}
