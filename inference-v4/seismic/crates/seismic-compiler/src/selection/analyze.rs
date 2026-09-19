use seismic_lang::family::{Family, Requirement};
use std::fmt;

/// `seismic analyze-search`: structure of the search, not a time prediction.
#[derive(Clone, Debug, Default)]
pub struct SearchAnalysis {
    pub templates: usize,
    pub occurrences: usize,
    pub choice_occurrences: usize,
    pub max_alternatives: usize,
    pub sites: usize,
    pub sequences: usize,
    pub intervals: usize,
    pub log10_raw_assignments: f64,
    pub independent_components: usize,
    pub obligations: usize,
}

pub fn analyze(family: &Family) -> SearchAnalysis {
    analyze_with(family, None)
}

/// As `analyze`, with the backend's interval count when it is known.
pub fn analyze_with(family: &Family, intervals: Option<usize>) -> SearchAnalysis {
    // Raw product: every choice times every site extent, ignoring guards and legality.
    let choices = family.occurrences.iter().filter(|o| o.candidates.len() > 1).map(|o| (o.candidates.len() as f64).log10());
    let extents = family.sites.iter().filter(|s| s.extent > 1).map(|s| (s.extent as f64).log10());
    let log10_raw_assignments = choices.chain(extents).sum();

    // Interaction graph over occurrences (all candidates of one occurrence are one
    // decision). A child interacts with its parent only when the parent is a real choice:
    // a fixed parent guards nothing. Structural references and requirements on sites owned
    // elsewhere join the two owners.
    let n = family.occurrences.len();
    let mut root: Vec<usize> = (0..n).collect();
    fn find(root: &mut [usize], mut i: usize) -> usize {
        while root[i] != i {
            root[i] = root[root[i]];
            i = root[i];
        }
        i
    }
    let mut join = |a: usize, b: usize| {
        if a < n && b < n {
            let (a, b) = (find(&mut root, a), find(&mut root, b));
            root[a] = b;
        }
    };
    let mut decides = vec![false; n];
    for (i, o) in family.occurrences.iter().enumerate() {
        decides[i] = o.candidates.len() > 1 || o.candidates.iter().any(|c| !c.sites.is_empty() || !c.sequences.is_empty());
        if let Some(parent) = o.parent {
            let p = parent.occurrence.0 as usize;
            if family.occurrences.get(p).is_some_and(|p| p.candidates.len() > 1) {
                join(i, p);
            }
        }
        for candidate in &o.candidates {
            let structural = candidate.structural.iter().map(|(_, r)| r.0);
            let required = candidate.requirements.iter().map(|r| match r {
                Requirement::Multiple { site, .. }
                | Requirement::AtLeast { site, .. }
                | Requirement::AtMost { site, .. }
                | Requirement::Equal { site, .. }
                | Requirement::Divides { site, .. } => *site,
            });
            for site in structural.chain(required) {
                if let Some(site) = family.sites.get(site.0 as usize) {
                    join(i, site.owner.occurrence.0 as usize);
                }
            }
        }
    }
    let mut components: Vec<usize> = (0..n).filter(|i| decides[*i]).map(|i| find(&mut root, i)).collect();
    components.sort_unstable();
    components.dedup();

    SearchAnalysis {
        templates: family.templates.len(),
        occurrences: n,
        choice_occurrences: family.occurrences.iter().filter(|o| o.candidates.len() > 1).count(),
        max_alternatives: family.occurrences.iter().map(|o| o.candidates.len()).max().unwrap_or(0),
        sites: family.sites.len(),
        sequences: family.sequences.len(),
        intervals: intervals.unwrap_or(0),
        log10_raw_assignments,
        independent_components: components.len(),
        obligations: family.obligations.len(),
    }
}

impl fmt::Display for SearchAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "templates {}  occurrences {} ({} with a choice, max {} alternatives)", self.templates, self.occurrences, self.choice_occurrences, self.max_alternatives)?;
        writeln!(f, "sites {}  sequences {}  intervals {}", self.sites, self.sequences, self.intervals)?;
        writeln!(f, "raw assignments <= 10^{:.2} (choices x site extents; an upper bound, not solve work)", self.log10_raw_assignments)?;
        write!(f, "independent components {}  obligations {}", self.independent_components, self.obligations)
    }
}
