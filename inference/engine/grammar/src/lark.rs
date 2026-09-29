//! Lark rendering of a lexical plan. Each Earley rule is rendered as
//! left-oriented boundary rules, so Earley-level sequences stay linear; each
//! terminal is defined once and referenced by name.
use crate::{
    lexeme::{EdgeKind, Plan},
    terminal::{Term, TermId, Terms},
};
use std::{collections::BTreeSet, fmt::Write};

fn repetition(min: u32, max: Option<u32>) -> String {
    match (min, max) {
        (1, Some(1)) => String::new(),
        (0, None) => "*".into(),
        (1, None) => "+".into(),
        (0, Some(1)) => "?".into(),
        (min, Some(max)) if min == max => format!("{{{min}}}"),
        (min, Some(max)) => format!("{{{min},{max}}}"),
        (min, None) => format!("{{{min},}}"),
    }
}

fn terminal(id: TermId) -> String {
    format!("T{id}")
}

fn state(rule: usize, boundary: u32) -> String {
    if boundary == 1 {
        format!("r{rule}")
    } else {
        format!("r{rule}_{boundary}")
    }
}

fn definition(term: &Term) -> String {
    match term {
        Term::Chars(set) => {
            let mut class = String::from("/[");
            for &(start, end) in set.ranges() {
                if start == end {
                    write!(class, "\\x{{{start:x}}}").unwrap();
                } else {
                    write!(class, "\\x{{{start:x}}}-\\x{{{end:x}}}").unwrap();
                }
            }
            class.push_str("]/");
            class
        }
        Term::Literal(text) => serde_json::to_string(text).expect("strings serialize"),
        Term::Seq(parts) if parts.is_empty() => "\"\"".into(),
        Term::Seq(parts) => parts
            .iter()
            .map(|&p| terminal(p))
            .collect::<Vec<_>>()
            .join(" "),
        Term::Alt(parts) => parts
            .iter()
            .map(|&p| terminal(p))
            .collect::<Vec<_>>()
            .join(" | "),
        Term::Repeat(part, min, max) => format!("{}{}", terminal(*part), repetition(*min, *max)),
        Term::NonEmpty(part) => format!("{} & /[\\x{{0}}-\\x{{10ffff}}]+/", terminal(*part)),
    }
}

fn operands(term: &Term) -> Vec<TermId> {
    match term {
        Term::Seq(parts) | Term::Alt(parts) => parts.clone(),
        Term::Repeat(part, _, _) | Term::NonEmpty(part) => vec![*part],
        Term::Chars(_) | Term::Literal(_) => Vec::new(),
    }
}

pub(crate) struct Rendered {
    pub lark: String,
    pub terminals: usize,
    pub lexemes: usize,
}

pub(crate) fn render(plan: &Plan, terms: &Terms) -> Rendered {
    let mut lark = String::from("%llguidance {}\nstart: r0\n");
    let mut lexemes = BTreeSet::new();
    for (index, rule) in plan.rules.iter().enumerate() {
        for target in 1..rule.boundaries {
            let alternatives = rule
                .edges
                .iter()
                .filter(|edge| edge.to == target)
                .map(|edge| {
                    let mut parts = Vec::new();
                    if edge.from != 0 {
                        parts.push(state(index, edge.from));
                    }
                    match edge.kind {
                        EdgeKind::Lexeme { lexeme, .. } => {
                            lexemes.insert(lexeme);
                            parts.push(terminal(lexeme));
                        }
                        EdgeKind::Call { rule, min, max } => {
                            parts.push(format!("r{rule}{}", repetition(min, max)))
                        }
                        EdgeKind::Eps => {}
                    }
                    if parts.is_empty() {
                        "\"\"".to_string()
                    } else {
                        parts.join(" ")
                    }
                })
                .collect::<Vec<_>>();
            if alternatives.is_empty() {
                continue;
            }
            writeln!(
                lark,
                "{}: {}",
                state(index, target),
                alternatives.join(" | ")
            )
            .unwrap();
        }
    }
    // Every terminal a lexeme depends on, in id order: operands precede the
    // terminals built from them.
    let mut used = BTreeSet::new();
    let mut pending = lexemes.iter().copied().collect::<Vec<_>>();
    while let Some(term) = pending.pop() {
        if used.insert(term) {
            pending.extend(operands(terms.get(term)));
        }
    }
    for &term in &used {
        writeln!(lark, "{}: {}", terminal(term), definition(terms.get(term))).unwrap();
    }
    Rendered {
        lark,
        terminals: used.len(),
        lexemes: lexemes.len(),
    }
}
