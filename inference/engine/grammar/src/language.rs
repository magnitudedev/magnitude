//! Emptiness questions over terminal languages, answered with derivre within
//! a per-compilation work budget. Planning and certification ask them.
use crate::terminal::{has_byte, Bytes, Term, TermId, Terms};
use llguidance::derivre::{raw::RelevanceCache, ExprRef, RegexAst, RegexBuilder};

/// Derivative work allowed for one emptiness question.
const FUEL: u64 = 200_000;
/// Derivative work allowed per compilation. A question left unanswered
/// counts as positive, which only makes the rendering finer.
const TOTAL_FUEL: u64 = 20_000_000;

/// derivre expressions for terminals and batched emptiness questions.
pub(crate) struct Regexes {
    builder: RegexBuilder,
    exprs: Vec<Option<ExprRef>>,
    fuel: u64,
    /// Questions asked, and those the budget left unanswered.
    pub questions: usize,
    pub exhausted: usize,
}

impl Regexes {
    pub(crate) fn new() -> Self {
        Self {
            builder: RegexBuilder::new(),
            exprs: Vec::new(),
            fuel: TOTAL_FUEL,
            questions: 0,
            exhausted: 0,
        }
    }
    fn byte_set(bytes: &Bytes) -> RegexAst {
        let mut words = vec![0u32; 8];
        for byte in 0..=255u8 {
            if has_byte(bytes, byte) {
                words[byte as usize / 32] |= 1 << (byte % 32);
            }
        }
        RegexAst::ByteSet(words)
    }
    fn any_bytes(min: u32) -> RegexAst {
        RegexAst::Repeat(Box::new(Self::byte_set(&[u64::MAX; 4])), min, u32::MAX)
    }
    /// Terminals are interned after their operands, so building in id
    /// order never recurses.
    fn expr(&mut self, terms: &Terms, id: TermId) -> ExprRef {
        while self.exprs.len() <= id as usize {
            self.exprs.push(None);
        }
        if let Some(expr) = self.exprs[id as usize] {
            return expr;
        }
        for current in 0..=id {
            if self.exprs[current as usize].is_some() {
                continue;
            }
            let child = |term: TermId, exprs: &[Option<ExprRef>]| {
                RegexAst::ExprRef(exprs[term as usize].expect("operands precede their terminal"))
            };
            let ast = match terms.get(current) {
                Term::Chars(set) => {
                    let mut class = String::from("[");
                    for &(start, end) in set.ranges() {
                        class.push_str(&format!("\\x{{{start:x}}}-\\x{{{end:x}}}"));
                    }
                    class.push(']');
                    RegexAst::Regex(class)
                }
                Term::Literal(text) => RegexAst::Literal(text.clone()),
                Term::Seq(parts) => {
                    RegexAst::Concat(parts.iter().map(|&p| child(p, &self.exprs)).collect())
                }
                Term::Alt(parts) => {
                    RegexAst::Or(parts.iter().map(|&p| child(p, &self.exprs)).collect())
                }
                Term::Repeat(part, min, max) => RegexAst::Repeat(
                    Box::new(child(*part, &self.exprs)),
                    *min,
                    max.unwrap_or(u32::MAX),
                ),
                Term::NonEmpty(part) => {
                    RegexAst::And(vec![child(*part, &self.exprs), Self::any_bytes(1)])
                }
            };
            let expr = self
                .builder
                .mk(&ast)
                .expect("terminal expressions are well formed");
            self.exprs[current as usize] = Some(expr);
        }
        self.exprs[id as usize].unwrap()
    }
    fn union(&mut self, terms: &Terms, parts: &[TermId]) -> RegexAst {
        RegexAst::Or(
            parts
                .iter()
                .map(|&part| RegexAst::ExprRef(self.expr(terms, part)))
                .collect(),
        )
    }
    /// Some text of `a` is also text of one of `others`.
    pub(crate) fn shared(&mut self, terms: &Terms, a: TermId, others: &[TermId]) -> ExprRef {
        let ast = RegexAst::And(vec![
            RegexAst::ExprRef(self.expr(terms, a)),
            self.union(terms, others),
        ]);
        self.builder.mk(&ast).expect("intersection of terminals")
    }
    /// Some text of `ended`, then a byte of `follow`, begins text of one of
    /// `continued`.
    pub(crate) fn overrun(
        &mut self,
        terms: &Terms,
        ended: TermId,
        follow: &Bytes,
        continued: &[TermId],
    ) -> ExprRef {
        let ast = RegexAst::And(vec![
            self.union(terms, continued),
            RegexAst::Concat(vec![
                RegexAst::ExprRef(self.expr(terms, ended)),
                Self::byte_set(follow),
                Self::any_bytes(0),
            ]),
        ]);
        self.builder.mk(&ast).expect("overrun of terminals")
    }
    /// Some text of `text` contains text of `delimiter`.
    pub(crate) fn contains(&mut self, terms: &Terms, text: TermId, delimiter: TermId) -> ExprRef {
        let ast = RegexAst::And(vec![
            RegexAst::ExprRef(self.expr(terms, text)),
            RegexAst::Concat(vec![
                Self::any_bytes(0),
                RegexAst::ExprRef(self.expr(terms, delimiter)),
                Self::any_bytes(0),
            ]),
        ]);
        self.builder.mk(&ast).expect("containment of terminals")
    }
    /// Non-emptiness of each expression; unanswerable questions count as
    /// non-empty.
    pub(crate) fn nonempty(&mut self, questions: &[ExprRef]) -> Vec<bool> {
        if questions.is_empty() {
            return Vec::new();
        }
        self.questions += questions.len();
        let mut exprs = self.builder.exprset().clone();
        let mut relevance = RelevanceCache::new();
        questions
            .iter()
            .map(|&question| {
                let answer = if self.fuel == 0 {
                    None
                } else {
                    let before = exprs.cost();
                    let answer = relevance
                        .is_non_empty_limited(&mut exprs, question, FUEL.min(self.fuel))
                        .ok();
                    self.fuel = self.fuel.saturating_sub(exprs.cost() - before);
                    answer
                };
                if answer.is_none() {
                    self.exhausted += 1;
                }
                answer.unwrap_or(true)
            })
            .collect()
    }
}
