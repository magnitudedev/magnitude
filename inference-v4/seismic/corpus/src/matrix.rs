//! The construct x context matrix (`scenarios/matrix/cells.txt`): which cells
//! require an executing scenario and which require a rejected one.
use std::fmt;
use std::path::Path;

/// One context column of the matrix, in file order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MatrixColumn {
    StraightLine,
    IfArm,
    Join,
    Ordered,
    Carry,
    ZeroTrip,
    Parallel,
    Argument,
    Result,
    Exit,
    AfterFailure,
    Async,
}

impl MatrixColumn {
    pub const ALL: [MatrixColumn; 12] = [
        Self::StraightLine,
        Self::IfArm,
        Self::Join,
        Self::Ordered,
        Self::Carry,
        Self::ZeroTrip,
        Self::Parallel,
        Self::Argument,
        Self::Result,
        Self::Exit,
        Self::AfterFailure,
        Self::Async,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::StraightLine => "sl",
            Self::IfArm => "if-arm",
            Self::Join => "join",
            Self::Ordered => "ordered",
            Self::Carry => "carry",
            Self::ZeroTrip => "zero-trip",
            Self::Parallel => "parallel",
            Self::Argument => "arg",
            Self::Result => "result",
            Self::Exit => "exit",
            Self::AfterFailure => "after-failure",
            Self::Async => "async",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|column| column.name() == name)
    }
}

/// A construct row. Rows are defined by `cells.txt`; a scenario tag naming an
/// unknown row is rejected by `tests/matrix.rs`, not by the header parser.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MatrixRow(String);

impl MatrixRow {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MatrixCell {
    pub row: MatrixRow,
    pub column: MatrixColumn,
}

impl MatrixCell {
    /// Parses `row/column`; `None` for any other spelling.
    pub fn parse(text: &str) -> Option<Self> {
        let (row, column) = text.split_once('/')?;
        if row.is_empty() || row.contains(char::is_whitespace) {
            return None;
        }
        Some(Self {
            row: MatrixRow::new(row),
            column: MatrixColumn::from_name(column)?,
        })
    }
}

impl fmt::Display for MatrixCell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.row.as_str(), self.column.name())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellMark {
    /// `+`: a scenario of class `Executes` is required.
    Executes,
    /// `!`: a scenario of class `Rejected` is required.
    Rejected,
    /// `.`: no scenario is required.
    Unrequired,
}

pub struct Matrix {
    rows: Vec<(MatrixRow, [CellMark; 12])>,
}

impl Matrix {
    pub fn load(path: &Path) -> Self {
        let text = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        Self::parse(&text)
    }

    /// Panics `cells.txt:<line>: <reason>` on anything but the matrix grammar:
    /// `#` comment lines, and `<row> <12 marks>` with an optional trailing comment.
    pub fn parse(text: &str) -> Self {
        let mut rows: Vec<(MatrixRow, [CellMark; 12])> = Vec::new();
        for (index, line) in text.lines().enumerate() {
            let fail = |reason: &str| -> ! { panic!("cells.txt:{}: {reason}", index + 1) };
            if line.trim().is_empty() || line.starts_with('#') {
                continue;
            }
            let cells = line.split_once('#').map_or(line, |(cells, _)| cells);
            let mut words = cells.split_whitespace();
            let row = MatrixRow::new(words.next().expect("non-empty line has a word"));
            let marks: Vec<CellMark> = words
                .map(|mark| match mark {
                    "+" => CellMark::Executes,
                    "!" => CellMark::Rejected,
                    "." => CellMark::Unrequired,
                    _ => fail(&format!("unknown mark `{mark}`")),
                })
                .collect();
            let Ok(marks) = <[CellMark; 12]>::try_from(marks) else {
                fail("a row has exactly 12 marks");
            };
            if rows.iter().any(|(known, _)| *known == row) {
                fail(&format!("row `{}` is declared twice", row.as_str()));
            }
            rows.push((row, marks));
        }
        Self { rows }
    }

    /// The mark of a cell, or `None` when the row is not in the matrix.
    pub fn mark(&self, cell: &MatrixCell) -> Option<CellMark> {
        self.rows
            .iter()
            .find(|(row, _)| *row == cell.row)
            .map(|(_, marks)| marks[cell.column as usize])
    }

    /// Every cell marked `+` or `!`, in file order.
    pub fn required(&self) -> impl Iterator<Item = (MatrixCell, CellMark)> + '_ {
        self.rows.iter().flat_map(|(row, marks)| {
            MatrixColumn::ALL
                .into_iter()
                .zip(marks)
                .filter(|(_, mark)| **mark != CellMark::Unrequired)
                .map(|(column, mark)| {
                    (
                        MatrixCell {
                            row: row.clone(),
                            column,
                        },
                        *mark,
                    )
                })
        })
    }
}
