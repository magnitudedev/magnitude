//! Two wires are postcard: the worker wire (`worker::protocol`) and the
//! checked bundle's checked section (`seismic_lang::bundle`). Postcard writes
//! no field names, so a decoder reads exactly the fields a type declares, in
//! order. A serde attribute that lets the written fields differ from the
//! declared ones (`skip_serializing_if`, a one-sided skip, `default` standing
//! in for a field never written, a `serialize_with` or `deserialize_with`
//! that names one side's function and leaves the other to the derive), or
//! that needs a self-describing format
//! (`flatten`, `untagged`, adjacent or internal tags, a hand-written impl or
//! `serialize_with`/`deserialize_with` function that asks
//! `is_human_readable` or reads `deserialize_any`), therefore corrupts or
//! refuses a message. No type either wire carries may hold one.
//!
//! The rule walks each wire's type closure from its roots (`HostMessage`,
//! `WorkerMessage`; `CheckedFields`, which a bundle writes, and
//! `CheckedParts`, which it reads) through the sources of every workspace
//! crate the engine depends on, which reach both: each field's, variant's,
//! `with`'s and `try_from`/`from`/`into`'s types, each resolved in the crate
//! its file imports it from (or re-exports it through). A field whose
//! `serialize_with` and `deserialize_with` name both sides' functions is
//! those functions': each body is judged as a hand-written impl is (that
//! each writes what the other reads is the crate's own tests' to show).
//! Impls a `macro_rules!` body writes for one of its arguments are
//! the hand-written impls of the type each invocation names. A type
//! argument the named type holds only in `PhantomData` is never on the wire
//! and is not walked. A name the walk cannot find is refused too, so a wire
//! type it cannot read fails here rather than going unchecked.

use proc_macro2::{Delimiter, TokenStream, TokenTree};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use syn::visit::Visit;

const REFUSED: [&str; 8] = [
    "skip_serializing",
    "skip_serializing_if",
    "skip_deserializing",
    "default",
    "flatten",
    "untagged",
    "tag",
    "content",
];

/// What a hand-written impl or serde function may not do: decide its form by
/// the format, or read whatever the input says it holds.
const SELF_DESCRIBING: [&str; 2] = ["is_human_readable", "deserialize_any"];

/// Names the walk does not look up: the language's and std's own, which
/// postcard reads positionally.
const STD: [&str; 37] = [
    "u8", "u16", "u32", "u64", "u128", "usize", "i8", "i16", "i32", "i64", "i128", "isize", "f32",
    "f64", "bool", "char", "str", "String", "Option", "Vec", "Box", "Arc", "Result", "BTreeMap",
    "BTreeSet", "HashMap", "HashSet", "VecDeque", "PathBuf", "Duration", "NonZeroU8", "NonZeroU16",
    "NonZeroU32", "NonZeroU64", "NonZeroUsize", "Range", "PhantomData",
];

/// Each wire's roots, by the crate that defines them.
const ROOTS: [(&str, &str); 4] = [
    ("magnitude_engine", "HostMessage"),
    ("magnitude_engine", "WorkerMessage"),
    ("seismic_lang", "CheckedFields"),
    ("seismic_lang", "CheckedParts"),
];

/// A source file's crate and the names its `use`s bring in, each by the
/// crate its path starts at.
struct Scope {
    file: PathBuf,
    krate: String,
    /// The crates the walk reads.
    known: Rc<BTreeSet<String>>,
    imports: BTreeMap<String, String>,
    globs: BTreeSet<String>,
}

impl Scope {
    /// The crate a path starting at `first` is in: the one it names, or
    /// this one (`crate`, `self`, `super` or a module of it).
    fn resolve(&self, first: &str) -> String {
        if self.known.contains(first) { first.to_owned() } else { self.krate.clone() }
    }

    /// Takes in the names `tree`, under a path starting at `first`, brings in.
    fn uses(&mut self, tree: &syn::UseTree, first: Option<&str>) {
        let start = |ident: &syn::Ident| first.map(str::to_owned).unwrap_or_else(|| ident.to_string());
        match tree {
            syn::UseTree::Path(path) => self.uses(&path.tree, Some(&start(&path.ident))),
            syn::UseTree::Name(name) => {
                let krate = self.resolve(&start(&name.ident));
                self.imports.insert(name.ident.to_string(), krate);
            }
            syn::UseTree::Rename(rename) => {
                let krate = self.resolve(&start(&rename.ident));
                self.imports.insert(rename.rename.to_string(), krate);
            }
            syn::UseTree::Glob(_) => {
                let krate = self.resolve(first.unwrap_or("self"));
                self.globs.insert(krate);
            }
            syn::UseTree::Group(group) => group.items.iter().for_each(|tree| self.uses(tree, first)),
        }
    }

    /// The crates `tree` names names in.
    fn named(&self, tree: &syn::UseTree) -> BTreeSet<String> {
        match tree {
            syn::UseTree::Path(syn::UsePath { ident, .. })
            | syn::UseTree::Name(syn::UseName { ident })
            | syn::UseTree::Rename(syn::UseRename { ident, .. }) => BTreeSet::from([self.resolve(&ident.to_string())]),
            syn::UseTree::Glob(_) => BTreeSet::from([self.krate.clone()]),
            syn::UseTree::Group(group) => group.items.iter().flat_map(|tree| self.named(tree)).collect(),
        }
    }

    /// The crates `path`, written in this file, may name an item of.
    fn crates(&self, path: &[String]) -> BTreeSet<String> {
        match path {
            [name] => match self.imports.get(name) {
                Some(krate) => BTreeSet::from([krate.clone()]),
                None => self.globs.iter().cloned().chain([self.krate.clone()]).collect(),
            },
            [first, ..] => BTreeSet::from([match self.imports.get(first) {
                Some(krate) if !self.known.contains(first) => krate.clone(),
                _ => self.resolve(first),
            }]),
            [] => unreachable!("a path has a segment"),
        }
    }
}

enum Definition {
    Struct(syn::ItemStruct),
    Enum(syn::ItemEnum),
    Alias(syn::ItemType),
}

/// Items of one kind by name, each with its file's scope.
type Named<T> = BTreeMap<String, Vec<(Rc<Scope>, T)>>;

/// A `macro_rules!` whose body writes a `Serialize` or `Deserialize` impl
/// for one of its arguments.
struct SerdeMacro {
    rules: Vec<Rule>,
}

struct Rule {
    matcher: Vec<TokenTree>,
    /// The impls the rule's body writes: the metavariable each implements
    /// for, and the impl's tokens.
    impls: Vec<(String, String)>,
}

#[derive(Default)]
struct Index {
    /// Every serde-derived type and alias.
    definitions: Named<Definition>,
    /// Hand-written `Serialize`/`Deserialize` impls, by the implementing
    /// type's name: the impl's tokens.
    impls: Named<String>,
    /// Every free function: its tokens, for a `serialize_with` or
    /// `deserialize_with` naming it.
    functions: Named<String>,
    /// Every struct, enum and alias, serde-derived or not: whether each of its
    /// generic parameters (types and consts, in order) is a type it holds
    /// outside `PhantomData`.
    shapes: Named<Vec<bool>>,
    /// The crates each crate's `pub use`s re-export from.
    reexports: BTreeMap<String, BTreeSet<String>>,
    /// Each crate's serde-writing `macro_rules!`, by name.
    macros: BTreeMap<(String, String), Vec<SerdeMacro>>,
    /// Item-level macro invocations: where, the macro's name, its tokens.
    invocations: Vec<(Rc<Scope>, String, TokenStream)>,
}

impl Index {
    fn read(&mut self, krate: &str, known: &Rc<BTreeSet<String>>, path: &Path) {
        let file = syn::parse_file(&std::fs::read_to_string(path).unwrap())
            .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        let mut scope = Scope {
            file: path.to_owned(),
            krate: krate.to_owned(),
            known: known.clone(),
            imports: BTreeMap::new(),
            globs: BTreeSet::new(),
        };
        let mut reexports = BTreeSet::new();
        uses(&file.items, &mut scope, &mut reexports);
        self.reexports.entry(krate.to_owned()).or_default().extend(reexports);
        self.items(&Rc::new(scope), &file.items);
    }

    fn items(&mut self, scope: &Rc<Scope>, items: &[syn::Item]) {
        let tokens = |item: &dyn quote::ToTokens| item.to_token_stream().to_string();
        for item in items {
            let mut define = |name: &syn::Ident, definition| {
                self.definitions
                    .entry(name.to_string())
                    .or_default()
                    .push((scope.clone(), definition));
            };
            match item {
                syn::Item::Struct(item) if !tested(&item.attrs) => {
                    let shape = shape(&item.generics, item.fields.iter().map(|field| &field.ty));
                    self.shapes.entry(item.ident.to_string()).or_default().push((scope.clone(), shape));
                    if derived(&item.attrs).any() {
                        define(&item.ident, Definition::Struct(item.clone()));
                    }
                }
                syn::Item::Enum(item) if !tested(&item.attrs) => {
                    let shape = shape(&item.generics, item.variants.iter().flat_map(|variant| &variant.fields).map(|field| &field.ty));
                    self.shapes.entry(item.ident.to_string()).or_default().push((scope.clone(), shape));
                    if derived(&item.attrs).any() {
                        define(&item.ident, Definition::Enum(item.clone()));
                    }
                }
                syn::Item::Type(item) if !tested(&item.attrs) => {
                    let shape = shape(&item.generics, std::iter::once(&*item.ty));
                    self.shapes.entry(item.ident.to_string()).or_default().push((scope.clone(), shape));
                    define(&item.ident, Definition::Alias(item.clone()))
                }
                syn::Item::Fn(item) if !tested(&item.attrs) => {
                    self.functions.entry(item.sig.ident.to_string()).or_default().push((scope.clone(), tokens(item)));
                }
                syn::Item::Mod(module) if !tested(&module.attrs) => {
                    if let Some((_, items)) = &module.content {
                        self.items(scope, items);
                    }
                }
                syn::Item::Impl(item) if !tested(&item.attrs) => {
                    let Some((_, trait_path, _)) = &item.trait_ else { continue };
                    let trait_name = trait_path.segments.last().unwrap().ident.to_string();
                    if trait_name != "Serialize" && trait_name != "Deserialize" {
                        continue;
                    }
                    if let syn::Type::Path(implementing) = &*item.self_ty {
                        let name = implementing.path.segments.last().unwrap().ident.to_string();
                        self.impls.entry(name).or_default().push((scope.clone(), tokens(item)));
                    }
                }
                syn::Item::Macro(item) if !tested(&item.attrs) => {
                    let name = item.mac.path.segments.last().unwrap().ident.to_string();
                    match &item.ident {
                        Some(defined) if name == "macro_rules" => {
                            if let Some(serde) = SerdeMacro::of(&item.mac.tokens, &scope.file) {
                                self.macros.entry((scope.krate.clone(), defined.to_string())).or_default().push(serde);
                            }
                        }
                        _ => self.invocations.push((scope.clone(), name, item.mac.tokens.clone())),
                    }
                }
                _ => {}
            }
        }
    }

    /// Records the impls each invocation of a serde-writing macro writes
    /// as the hand-written impls of the type it names. Run once every
    /// source is read, since a macro may be defined after it is invoked.
    fn expand(&mut self) {
        let mut written = Vec::new();
        for (scope, name, tokens) in &self.invocations {
            let Some(macros) = self.macros.get(&(scope.krate.clone(), name.clone())) else { continue };
            let input: Vec<TokenTree> = tokens.clone().into_iter().collect();
            for serde in macros {
                let (bindings, rule) = serde
                    .rules
                    .iter()
                    .find_map(|rule| matches(&rule.matcher, &input).map(|bindings| (bindings, rule)))
                    .unwrap_or_else(|| panic!("{}: `{name}!({tokens})` matches none of its rules", scope.file.display()));
                for (variable, impl_tokens) in &rule.impls {
                    let implementing = match bindings.get(variable).map(Vec::as_slice) {
                        Some([TokenTree::Ident(ident)]) => ident.to_string(),
                        bound => panic!("{}: `{name}!` implements serde for `${variable}`, bound to {bound:?}, not a name", scope.file.display()),
                    };
                    written.push((implementing, scope.clone(), impl_tokens.clone()));
                }
            }
        }
        for (implementing, scope, tokens) in written {
            self.impls.entry(implementing).or_default().push((scope, tokens));
        }
    }

    /// Each layer of crates a name written as reaching `crates` is looked
    /// up in: `crates`, then the crates those re-export from, nearest first.
    fn layers(&self, crates: BTreeSet<String>) -> Vec<BTreeSet<String>> {
        let (mut layers, mut seen, mut layer) = (Vec::new(), crates.clone(), crates);
        while !layer.is_empty() {
            let next = layer
                .iter()
                .flat_map(|krate| self.reexports.get(krate).into_iter().flatten())
                .filter(|krate| seen.insert((*krate).clone()))
                .cloned()
                .collect();
            layers.push(std::mem::replace(&mut layer, next));
        }
        layers
    }

    /// `name`'s definitions and hand-written impls in the nearest layer
    /// that has any.
    fn lookup(&self, name: &str, crates: BTreeSet<String>) -> (Vec<&(Rc<Scope>, Definition)>, Vec<&(Rc<Scope>, String)>) {
        self.layers(crates)
            .into_iter()
            .map(|layer| (within(&self.definitions, name, &layer), within(&self.impls, name, &layer)))
            .find(|(definitions, impls)| !definitions.is_empty() || !impls.is_empty())
            .unwrap_or_default()
    }

    /// `name`'s items of one kind in the nearest layer that has any.
    fn nearest<'a, T>(&self, items: &'a Named<T>, name: &str, crates: BTreeSet<String>) -> Vec<&'a (Rc<Scope>, T)> {
        self.layers(crates)
            .into_iter()
            .map(|layer| within(items, name, &layer))
            .find(|found| !found.is_empty())
            .unwrap_or_default()
    }
}

/// `name`'s items of one kind in `crates`.
fn within<'a, T>(items: &'a Named<T>, name: &str, crates: &BTreeSet<String>) -> Vec<&'a (Rc<Scope>, T)> {
    items.get(name).into_iter().flatten().filter(|(of, _)| crates.contains(&of.krate)).collect()
}

impl SerdeMacro {
    /// The serde impls `macro_rules! { tokens }` writes, if it writes any.
    fn of(tokens: &TokenStream, file: &Path) -> Option<Self> {
        let tokens: Vec<TokenTree> = tokens.clone().into_iter().collect();
        let mut rules = Vec::new();
        for rule in tokens.split(|token| matches!(token, TokenTree::Punct(punct) if punct.as_char() == ';')) {
            match rule {
                [] => {}
                [TokenTree::Group(matcher), TokenTree::Punct(equals), TokenTree::Punct(arrow), TokenTree::Group(body)]
                    if equals.as_char() == '=' && arrow.as_char() == '>' =>
                {
                    rules.push(Rule { matcher: matcher.stream().into_iter().collect(), impls: serde_impls(body.stream()) })
                }
                _ => panic!("{}: a macro_rules! rule is `(matcher) => {{ body }}`", file.display()),
            }
        }
        rules.iter().any(|rule| !rule.impls.is_empty()).then_some(Self { rules })
    }
}

/// The `impl ... Serialize|Deserialize ... for $variable { ... }` items of a
/// macro body: each one's metavariable and tokens.
fn serde_impls(body: TokenStream) -> Vec<(String, String)> {
    let body: Vec<TokenTree> = body.into_iter().collect();
    let mut impls = Vec::new();
    for (start, token) in body.iter().enumerate() {
        if !matches!(token, TokenTree::Ident(ident) if ident == "impl") {
            continue;
        }
        let Some(end) = body[start..].iter().position(|token| matches!(token, TokenTree::Group(group) if group.delimiter() == Delimiter::Brace)) else {
            continue;
        };
        let header = &body[start..start + end];
        let Some(at) = header.iter().position(|token| matches!(token, TokenTree::Ident(ident) if ident == "for")) else { continue };
        let serde = header[..at]
            .iter()
            .any(|token| matches!(token, TokenTree::Ident(ident) if ident == "Serialize" || ident == "Deserialize"));
        if let (true, [TokenTree::Punct(dollar), TokenTree::Ident(variable), ..]) = (serde, &header[at + 1..]) {
            if dollar.as_char() == '$' {
                let tokens: TokenStream = body[start..=start + end].iter().cloned().collect();
                impls.push((variable.to_string(), tokens.to_string()));
            }
        }
    }
    impls
}

/// What a match binds each metavariable outside a repetition to.
type Bindings = BTreeMap<String, Vec<TokenTree>>;

/// Matches all of `input` against a `macro_rules!` matcher, as rustc does
/// for the fragments item-level macros here take.
fn matches(matcher: &[TokenTree], input: &[TokenTree]) -> Option<Bindings> {
    let Some((first, rest)) = matcher.split_first() else { return input.is_empty().then(Bindings::new) };
    match (first, rest) {
        (TokenTree::Punct(dollar), [TokenTree::Ident(variable), TokenTree::Punct(colon), TokenTree::Ident(fragment), rest @ ..])
            if dollar.as_char() == '$' && colon.as_char() == ':' =>
        {
            let lengths = match fragment.to_string().as_str() {
                "ident" | "tt" | "literal" => 1..=1,
                "lifetime" => 2..=2,
                "vis" => 0..=input.len(),
                "meta" | "ty" | "expr" | "path" | "pat" | "pat_param" | "item" | "block" | "stmt" => 1..=input.len(),
                fragment => panic!("macro fragment `{fragment}` is not one the walk matches"),
            };
            lengths.filter(|&length| length <= input.len()).find_map(|length| {
                let mut bindings = matches(rest, &input[length..])?;
                bindings.insert(variable.to_string(), input[..length].to_vec());
                Some(bindings)
            })
        }
        (TokenTree::Punct(dollar), [TokenTree::Group(repeated), rest @ ..])
            if dollar.as_char() == '$' && repeated.delimiter() == Delimiter::Parenthesis =>
        {
            let operator = |token: &TokenTree| match token {
                TokenTree::Punct(punct) if matches!(punct.as_char(), '*' | '+' | '?') => Some(punct.as_char()),
                _ => None,
            };
            let (separator, operator, rest) = match rest {
                [op, rest @ ..] if operator(op).is_some() => (None, operator(op).unwrap(), rest),
                [separator, op, rest @ ..] if operator(op).is_some() => (Some(separator), operator(op).unwrap(), rest),
                _ => panic!("a macro repetition ends in `*`, `+` or `?`"),
            };
            let inner: Vec<TokenTree> = repeated.stream().into_iter().collect();
            repeat(&inner, separator, operator, rest, input, 0)
        }
        (TokenTree::Group(group), _) => {
            let Some((TokenTree::Group(given), tail)) = input.split_first() else { return None };
            if given.delimiter() != group.delimiter() {
                return None;
            }
            let contents: Vec<TokenTree> = group.stream().into_iter().collect();
            let given: Vec<TokenTree> = given.stream().into_iter().collect();
            let mut bindings = matches(&contents, &given)?;
            bindings.extend(matches(rest, tail)?);
            Some(bindings)
        }
        (literal, _) => {
            let (given, tail) = input.split_first()?;
            (!matches!(given, TokenTree::Group(_)) && given.to_string() == literal.to_string())
                .then(|| matches(rest, tail))
                .flatten()
        }
    }
}

/// Matches `input` against `done` repetitions of `inner` so far, then any
/// more `operator` admits, then `rest`. Bindings inside a repetition are
/// not single values and are not kept.
fn repeat(inner: &[TokenTree], separator: Option<&TokenTree>, operator: char, rest: &[TokenTree], input: &[TokenTree], done: usize) -> Option<Bindings> {
    let stopped = (operator != '+' || done > 0).then(|| matches(rest, input)).flatten();
    if stopped.is_some() || (operator == '?' && done == 1) {
        return stopped;
    }
    let input = match (done, separator) {
        (0, _) | (_, None) => input,
        (_, Some(separator)) => match input.split_first() {
            Some((given, tail)) if given.to_string() == separator.to_string() => tail,
            _ => return None,
        },
    };
    (1..=input.len()).find_map(|length| {
        matches(inner, &input[..length])?;
        repeat(inner, separator, operator, rest, &input[length..], done + 1)
    })
}

/// Collects the `use`s of `items` and their inline modules into `scope`, and
/// the crates its `pub use`s name into `reexports`.
fn uses(items: &[syn::Item], scope: &mut Scope, reexports: &mut BTreeSet<String>) {
    for item in items {
        match item {
            syn::Item::Use(item) if !tested(&item.attrs) => {
                scope.uses(&item.tree, None);
                if !matches!(item.vis, syn::Visibility::Inherited) {
                    reexports.extend(scope.named(&item.tree));
                }
            }
            syn::Item::Mod(module) if !tested(&module.attrs) => {
                if let Some((_, items)) = &module.content {
                    uses(items, scope, reexports);
                }
            }
            _ => {}
        }
    }
}

fn tested(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|attr| {
        attr.path().is_ident("cfg")
            && attr.meta.require_list().is_ok_and(|list| list.tokens.to_string() == "test")
    })
}

/// Which serde traits an item derives.
#[derive(Clone, Copy)]
struct Derived {
    serialize: bool,
    deserialize: bool,
}

impl Derived {
    /// Only a type serde derives for is a wire type's definition: a
    /// namesake that does not serialize is not one.
    fn any(self) -> bool {
        self.serialize || self.deserialize
    }
}

fn derived(attrs: &[syn::Attribute]) -> Derived {
    let mut derived = Derived { serialize: false, deserialize: false };
    for attr in attrs.iter().filter(|attr| attr.path().is_ident("derive")) {
        let paths = attr
            .parse_args_with(syn::punctuated::Punctuated::<syn::Path, syn::Token![,]>::parse_terminated)
            .unwrap_or_else(|error| panic!("a derive lists paths: {error}"));
        for path in paths {
            match path.segments.last().unwrap().ident.to_string().as_str() {
                "Serialize" => derived.serialize = true,
                "Deserialize" => derived.deserialize = true,
                _ => {}
            }
        }
    }
    derived
}

/// Whether each of `generics`' type and const parameters, in order, is a
/// type `types` hold outside `PhantomData` (a const is never a type).
fn shape<'a>(generics: &syn::Generics, types: impl Iterator<Item = &'a syn::Type>) -> Vec<bool> {
    let mut paths = Paths::default();
    types.for_each(|ty| paths.visit_type(ty));
    let all = paths.all();
    let held = |name: &syn::Ident| all.iter().any(|path| path.len() == 1 && *name == path[0]);
    generics
        .params
        .iter()
        .filter_map(|param| match param {
            syn::GenericParam::Type(param) => Some(held(&param.ident)),
            syn::GenericParam::Const(_) => Some(false),
            syn::GenericParam::Lifetime(_) => None,
        })
        .collect()
}

/// A type path a type mentions: its segments' names, and the generic
/// arguments of its last segment, which the walk follows only where the
/// type the path names holds them.
struct Mention {
    path: Vec<String>,
    arguments: Vec<syn::Type>,
}

/// Every type path a type mentions, its last segment's generic arguments
/// kept apart.
#[derive(Default)]
struct Paths(Vec<Mention>);

impl Paths {
    /// Every path, the arguments' own included.
    fn all(&self) -> Vec<Vec<String>> {
        let mut all = Vec::new();
        for mention in &self.0 {
            let mut inner = Paths::default();
            mention.arguments.iter().for_each(|ty| inner.visit_type(ty));
            all.push(mention.path.clone());
            all.extend(inner.all());
        }
        all
    }
}

impl<'a> Visit<'a> for Paths {
    fn visit_path(&mut self, path: &'a syn::Path) {
        let segments: Vec<&syn::PathSegment> = path.segments.iter().collect();
        let (last, leading) = segments.split_last().expect("a path has a segment");
        leading.iter().for_each(|segment| self.visit_path_arguments(&segment.arguments));
        let mut arguments = Vec::new();
        match &last.arguments {
            syn::PathArguments::AngleBracketed(angled) => {
                for argument in &angled.args {
                    match argument {
                        syn::GenericArgument::Type(ty) => arguments.push(ty.clone()),
                        // A `PhantomData` holds nothing: its own argument is never on the wire.
                        argument if last.ident != "PhantomData" => self.visit_generic_argument(argument),
                        _ => {}
                    }
                }
            }
            arguments_of_function => self.visit_path_arguments(arguments_of_function),
        }
        if last.ident == "PhantomData" {
            arguments.clear();
        }
        self.0.push(Mention { path: path.segments.iter().map(|segment| segment.ident.to_string()).collect(), arguments });
    }
}

/// A path the walk resolves: where, and the type it is written in.
struct Pending {
    mention: Mention,
    crates: BTreeSet<String>,
    scope: Option<Rc<Scope>>,
    from: String,
    /// The generic parameters of the item the path is written in.
    parameters: Rc<BTreeSet<String>>,
}

/// The paths `types`, written in `scope` within `from`, mention.
fn mentions(types: &[syn::Type], scope: &Rc<Scope>, from: &str, parameters: &Rc<BTreeSet<String>>) -> Vec<Pending> {
    let mut paths = Paths::default();
    types.iter().for_each(|ty| paths.visit_type(ty));
    paths
        .0
        .into_iter()
        .filter(|mention| !(mention.path.len() == 1 && (parameters.contains(&mention.path[0]) || mention.path[0] == "Self")))
        .map(|mention| Pending {
            crates: scope.crates(&mention.path),
            mention,
            scope: Some(scope.clone()),
            from: from.to_owned(),
            parameters: parameters.clone(),
        })
        .collect()
}

/// The engine's workspace dependencies, by library name, and their source
/// directories: every workspace package reached from this one by normal
/// path dependencies, as cargo reads the manifests (`--no-deps`, so nothing
/// is resolved or fetched).
fn sources() -> Vec<(String, PathBuf)> {
    let package = std::env::var_os("CARGO_MANIFEST_DIR").expect("cargo runs the test from its package");
    let cargo = std::env::var_os("CARGO").expect("cargo names itself to the test");
    let output = std::process::Command::new(cargo)
        .args(["metadata", "--format-version", "1", "--no-deps", "--offline"])
        .current_dir(&package)
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let metadata: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    // Each workspace package by its directory.
    let packages: BTreeMap<PathBuf, &serde_json::Value> = metadata["packages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|package| (Path::new(package["manifest_path"].as_str().unwrap()).parent().unwrap().to_owned(), package))
        .collect();
    let engine = PathBuf::from(&package);
    let (mut reached, mut pending) = (BTreeSet::from([engine.clone()]), vec![engine]);
    while let Some(directory) = pending.pop() {
        for dependency in packages[&directory]["dependencies"].as_array().unwrap() {
            let Some(path) = dependency["path"].as_str() else { continue };
            if dependency["kind"].is_null() && reached.insert(PathBuf::from(path)) {
                pending.push(PathBuf::from(path));
            }
        }
    }
    reached
        .into_iter()
        .map(|directory| {
            let package = packages
                .get(&directory)
                .unwrap_or_else(|| panic!("{} is a workspace package", directory.display()));
            let library = package["targets"]
                .as_array()
                .unwrap()
                .iter()
                .find(|target| target["kind"].as_array().unwrap().iter().any(|kind| kind == "lib"))
                .unwrap_or_else(|| panic!("{} is a library", directory.display()));
            let source = Path::new(library["src_path"].as_str().unwrap()).parent().unwrap().to_owned();
            (library["name"].as_str().unwrap().replace('-', "_"), source)
        })
        .collect()
}

#[test]
fn wire_types_carry_no_serde_form_postcard_cannot_read() {
    let mut index = Index::default();
    let crates = sources();
    let known: Rc<BTreeSet<String>> = Rc::new(crates.iter().map(|(krate, _)| krate.clone()).collect());
    for (krate, source) in crates {
        let mut pending = vec![source];
        while let Some(path) = pending.pop() {
            if path.is_dir() {
                pending.extend(std::fs::read_dir(&path).unwrap().map(|entry| entry.unwrap().path()));
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                index.read(&krate, &known, &path);
            }
        }
    }
    index.expand();

    // Each reached type (its file and name) and the type it was reached from.
    let mut reached = BTreeMap::<(PathBuf, String), String>::new();
    let mut walk: Vec<Pending> = ROOTS
        .iter()
        .map(|&(krate, root)| Pending {
            mention: Mention { path: vec![root.to_owned()], arguments: Vec::new() },
            crates: BTreeSet::from([krate.to_owned()]),
            scope: None,
            from: String::new(),
            parameters: Rc::default(),
        })
        .collect();
    // Each hand-written impl and serde function judged, by its file and tokens.
    let mut checked = BTreeSet::new();
    let (mut offenders, mut unfound) = (Vec::new(), Vec::new());
    let mut judge = |of: &Rc<Scope>, tokens: &String, what: String, offenders: &mut Vec<String>| {
        if checked.insert((of.file.clone(), tokens.clone())) {
            for refused in SELF_DESCRIBING.iter().filter(|refused| tokens.contains(*refused)) {
                offenders.push(format!("{}: {what}: `{refused}`", of.file.display()));
            }
        }
    };
    while let Some(Pending { mention, crates, scope, from, parameters }) = walk.pop() {
        let name = mention.path.last().unwrap().clone();
        // The arguments the named type holds: all of them for a type the
        // walk does not read, or that it finds no shape of. A namesake with
        // no parameter at a position is not the type named and has no say.
        let holds = |position: usize| {
            let shapes = index.nearest(&index.shapes, &name, crates.clone());
            let shapes: Vec<bool> = shapes.iter().filter_map(|(_, shape)| shape.get(position).copied()).collect();
            shapes.is_empty() || shapes.contains(&true)
        };
        if let Some(scope) = &scope {
            let held: Vec<syn::Type> = mention
                .arguments
                .iter()
                .enumerate()
                .filter(|&(position, _)| STD.contains(&name.as_str()) || holds(position))
                .map(|(_, ty)| ty.clone())
                .collect();
            walk.extend(mentions(&held, scope, &from, &parameters));
        }
        if STD.contains(&name.as_str()) {
            continue;
        }
        let (definitions, impls) = index.lookup(&name, crates);
        if definitions.is_empty() && impls.is_empty() {
            unfound.push(format!("{} in {from}", mention.path.join("::")));
            continue;
        }
        for (of, tokens) in &impls {
            judge(of, tokens, format!("{name}'s impl"), &mut offenders);
        }
        for (of, definition) in definitions {
            if reached.insert((of.file.clone(), name.clone()), from.clone()).is_some() {
                continue;
            }
            let file = of.file.as_path();
            let mut types = Vec::new();
            let mut functions = Vec::new();
            let generics = match definition {
                Definition::Struct(item) => {
                    let derive = derived(&item.attrs);
                    serde(file, &name, &item.attrs, &mut types, &mut offenders);
                    fields(file, &name, derive, &item.fields, &mut types, &mut functions, &mut offenders);
                    &item.generics
                }
                Definition::Enum(item) => {
                    let derive = derived(&item.attrs);
                    serde(file, &name, &item.attrs, &mut types, &mut offenders);
                    for variant in &item.variants {
                        match serde(file, &name, &variant.attrs, &mut types, &mut offenders) {
                            Form::Skipped => {}
                            Form::With(path) => types.push(syn::Type::Path(syn::TypePath { qself: None, path })),
                            Form::Encoded { serialize, deserialize } => {
                                let derive = Derived {
                                    serialize: derive.serialize && serialize.is_none(),
                                    deserialize: derive.deserialize && deserialize.is_none(),
                                };
                                functions.extend(serialize.into_iter().chain(deserialize));
                                fields(file, &name, derive, &variant.fields, &mut types, &mut functions, &mut offenders);
                            }
                        }
                    }
                    &item.generics
                }
                Definition::Alias(item) => {
                    types.push((*item.ty).clone());
                    &item.generics
                }
            };
            let within = format!("{name} ({})", file.display());
            for function in functions {
                let path: Vec<String> = function.segments.iter().map(|segment| segment.ident.to_string()).collect();
                let found = index.nearest(&index.functions, path.last().unwrap(), of.crates(&path));
                if found.is_empty() {
                    unfound.push(format!("{} in {within}", path.join("::")));
                }
                for (at, tokens) in found {
                    judge(at, tokens, format!("{name}'s `{}`", path.join("::")), &mut offenders);
                }
            }
            let parameters: Rc<BTreeSet<String>> = Rc::new(generics.type_params().map(|param| param.ident.to_string()).collect());
            walk.extend(mentions(&types, of, &within, &parameters));
        }
    }
    let names: BTreeSet<&str> = reached.keys().map(|(_, name)| name.as_str()).collect();
    assert!(unfound.is_empty(), "wire types the walk cannot find: {unfound:#?}");
    assert!(names.contains("PoolNorm"), "the walk follows the execution manifest to the model definition");
    assert!(names.contains("ExprKind"), "the walk follows the checked section to the checked IR");
    assert!(offenders.is_empty(), "serde forms postcard cannot read: {offenders:#?}");
}

/// Takes in what `fields`, of an item deriving `derive`, put on the wire:
/// the types to walk and the serde functions to judge.
fn fields(file: &Path, name: &str, derive: Derived, fields: &syn::Fields, types: &mut Vec<syn::Type>, functions: &mut Vec<syn::Path>, offenders: &mut Vec<String>) {
    for field in fields {
        serde(file, name, &field.attrs, types, offenders).place(derive, &field.ty, types, functions);
    }
}

/// What an item's serde attributes make of its declared form.
enum Form {
    /// `skip`: neither written nor read.
    Skipped,
    /// `with`: the path's form (a remote derive) on both sides.
    With(syn::Path),
    /// Each side by the function its `serialize_with`/`deserialize_with`
    /// names, or by the derive where none does (a one-sided pair is refused).
    Encoded { serialize: Option<syn::Path>, deserialize: Option<syn::Path> },
}

impl Form {
    /// Takes in what a field of declared type `ty`, in an item deriving
    /// `derive`, puts on the wire: the types to walk and the functions to
    /// judge.
    fn place(self, derive: Derived, ty: &syn::Type, types: &mut Vec<syn::Type>, functions: &mut Vec<syn::Path>) {
        match self {
            Form::Skipped => {}
            Form::With(path) => types.push(syn::Type::Path(syn::TypePath { qself: None, path })),
            Form::Encoded { serialize, deserialize } => {
                if (derive.serialize && serialize.is_none()) || (derive.deserialize && deserialize.is_none()) {
                    types.push(ty.clone());
                }
                functions.extend(serialize.into_iter().chain(deserialize));
            }
        }
    }
}

/// Checks `attrs`' serde attributes, those a `cfg_attr` applies included,
/// pushing the types a `try_from`, `from` or `into` names in place of the
/// item's own form.
fn serde(file: &Path, name: &str, attrs: &[syn::Attribute], types: &mut Vec<syn::Type>, offenders: &mut Vec<String>) -> Form {
    let (mut skipped, mut with, mut serialize, mut deserialize) = (false, None, None, None);
    let mut check = |meta: syn::meta::ParseNestedMeta| -> syn::Result<()> {
        let key = meta.path.get_ident().map(ToString::to_string).unwrap_or_default();
        if meta.input.peek(syn::token::Paren) {
            // A nested list (`rename(...)`, `bound(...)`): nothing it holds is refused.
            return meta.parse_nested_meta(|nested| {
                if let Ok(value) = nested.value() {
                    value.parse::<syn::LitStr>()?;
                }
                Ok(())
            });
        }
        let value = meta.value().ok().map(|value| value.parse::<syn::LitStr>()).transpose()?;
        if REFUSED.contains(&key.as_str()) {
            offenders.push(format!("{}: {name}: `{key}`", file.display()));
        }
        match (key.as_str(), value) {
            ("skip", _) => skipped = true,
            ("with", Some(value)) => with = Some(value.parse::<syn::Path>()?),
            ("serialize_with", Some(value)) => serialize = Some(value.parse::<syn::Path>()?),
            ("deserialize_with", Some(value)) => deserialize = Some(value.parse::<syn::Path>()?),
            ("try_from" | "from" | "into", Some(value)) => types.push(value.parse()?),
            _ => {}
        }
        Ok(())
    };
    for attr in attrs {
        let lists: Vec<syn::MetaList> = if attr.path().is_ident("serde") {
            vec![attr.meta.require_list().cloned().unwrap_or_else(|error| panic!("{}: {name}: {error}", file.display()))]
        } else if attr.path().is_ident("cfg_attr") {
            // `cfg_attr(predicate, attr, ...)`: the serde attributes it applies.
            attr.parse_args_with(syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated)
                .unwrap_or_else(|error| panic!("{}: {name}: {error}", file.display()))
                .into_iter()
                .skip(1)
                .filter_map(|meta| match meta {
                    syn::Meta::List(list) if list.path.is_ident("serde") => Some(list),
                    _ => None,
                })
                .collect()
        } else {
            Vec::new()
        };
        for list in lists {
            list.parse_nested_meta(&mut check).unwrap_or_else(|error| panic!("{}: {name}: {error}", file.display()));
        }
    }
    // A function for one side beside the derive for the other lets the written
    // form differ from the read one, and nothing here can compare the two.
    if serialize.is_some() != deserialize.is_some() {
        let named = if serialize.is_some() { "serialize_with" } else { "deserialize_with" };
        offenders.push(format!("{}: {name}: a one-sided `{named}`", file.display()));
    }
    // serde itself refuses `with` beside either function, and a skip beside any of them.
    match (skipped, with) {
        (true, _) => Form::Skipped,
        (false, Some(path)) => Form::With(path),
        (false, None) => Form::Encoded { serialize, deserialize },
    }
}
