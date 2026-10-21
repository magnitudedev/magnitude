//! The proposal budget shared by sequential and block drafters.

use magnitude_executor::draft_ladder;

/// A round's proposal budget, excluding its anchor and target bonus token:
/// the proposals the drafter generates and the prefix of them offered to
/// verification.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DraftBudget {
    pub generated: usize,
    pub verified: usize,
}

/// How a load's rounds take their proposal budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DraftPolicy {
    /// Every round takes this budget.
    Fixed(DraftBudget),
    /// Each round's budget is chosen from what the load's rounds measure,
    /// among the widths the load prepared up to its proposal width.
    Adaptive,
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

    /// The proposals a round generates when it may produce at most `limit`:
    /// the widest width of the budget's ladder ([`draft_ladder`], what a
    /// load of this budget prepares) that fits, none when none does. Away
    /// from a request's end that is the budget's own generated count.
    pub fn generated_within(&self, limit: usize) -> usize {
        draft_ladder(self.generated)
            .into_iter()
            .rev()
            .find(|width| *width <= limit)
            .unwrap_or(0)
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

    #[test]
    fn a_round_near_the_end_drafts_the_widest_prepared_width_that_fits() {
        let budget = DraftBudget::new(7, 7).unwrap();
        // The ladder of 7 is 1, 3 and 7.
        let widths = (0..=9)
            .map(|limit| budget.generated_within(limit))
            .collect::<Vec<_>>();
        assert_eq!(widths, [0, 1, 1, 3, 3, 3, 3, 7, 7, 7]);
        // A width off the power ladder is its own widest rung.
        let budget = DraftBudget::new(5, 2).unwrap();
        let widths = (0..=6)
            .map(|limit| budget.generated_within(limit))
            .collect::<Vec<_>>();
        assert_eq!(widths, [0, 1, 1, 3, 3, 5, 5]);
        // A budget drafting nothing never drafts.
        assert_eq!(DraftBudget::new(0, 0).unwrap().generated_within(8), 0);
    }
}
