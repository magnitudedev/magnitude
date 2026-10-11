//! The proposal budget shared by sequential and block drafters.

/// A round's proposal budget, excluding its anchor and target bonus token:
/// the proposals the drafter generates and the prefix of them offered to
/// verification.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DraftBudget {
    pub generated: usize,
    pub verified: usize,
}

impl DraftBudget {
    pub fn new(generated: usize, verified: usize) -> Result<Self, String> {
        if verified > generated || generated > u8::MAX as usize {
            return Err("draft budget requires verified <= generated <= 255".into());
        }
        Ok(Self {
            generated,
            verified,
        })
    }

    /// Whether a method drafting at most `cap` proposals can serve this
    /// budget. A budget that generates proposals must offer at least one.
    pub fn validate(&self, cap: usize) -> Result<(), String> {
        if self.verified <= self.generated
            && self.generated <= cap.min(u8::MAX as usize)
            && (self.generated == 0 || self.verified > 0)
        {
            Ok(())
        } else {
            Err("draft budget is invalid or exceeds the method's proposal width".into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DraftBudget;

    #[test]
    fn construction_bounds_the_verified_prefix_and_the_width() {
        assert_eq!(
            DraftBudget::new(3, 1),
            Ok(DraftBudget {
                generated: 3,
                verified: 1
            })
        );
        assert!(DraftBudget::new(0, 0).is_ok());
        assert!(DraftBudget::new(255, 255).is_ok());
        assert!(DraftBudget::new(1, 2).is_err());
        assert!(DraftBudget::new(256, 1).is_err());
    }

    #[test]
    fn validation_bounds_a_budget_by_the_method_width() {
        let budget = |generated, verified| DraftBudget {
            generated,
            verified,
        };
        assert!(budget(3, 3).validate(3).is_ok());
        assert!(budget(3, 1).validate(3).is_ok());
        assert!(budget(2, 1).validate(3).is_ok());
        // A budget drafting nothing is plain generation under a draft method.
        assert!(budget(0, 0).validate(3).is_ok());
        assert!(budget(4, 1).validate(3).is_err());
        assert!(budget(3, 4).validate(4).is_err());
        // Generated proposals that are never offered are wasted drafter work.
        assert!(budget(3, 0).validate(3).is_err());
        // Fields are public and deserialized: the width bound holds here too.
        assert!(budget(256, 1).validate(usize::MAX).is_err());
    }
}
