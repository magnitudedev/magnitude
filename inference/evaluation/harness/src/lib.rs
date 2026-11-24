//! Shared host-only template rendering and run-record primitives used by both
//! `magnitude-harness` and the lightweight `magnitude-render` binary.

pub mod artifacts;
pub mod digest;
pub mod record;
pub mod render;
pub mod template;

use artifacts::ArtifactsFile;
use clap::Args;
use magnitude_engine::host::HostArtifacts;
use magnitude_engine::options::{PackageOptions, ProjectorSelection};
use record::{Run, RunRecord};
use render::{Rendered, Renderer, RequestOptions};
use serde_json::{Map, Value};
use std::path::PathBuf;
use template::TemplateFile;

#[derive(Args)]
pub struct Common {
    /// The catalog acquisition object of the model (JSON).
    #[arg(long, value_name = "FILE")]
    pub artifacts: PathBuf,
    /// A template file (repeatable): its lanes, sized to its context.
    #[arg(long = "template", value_name = "FILE", required = true)]
    pub templates: Vec<PathBuf>,
    /// Chat-completions request fields every rendered request carries, as
    /// the serving benchmark sends them (a JSON object); by default
    /// `{"chat_template_kwargs": {"enable_thinking": false}}`.
    #[arg(long, value_name = "JSON")]
    pub request_options: Option<String>,
}

impl Common {
    pub fn request_options(&self) -> Result<RequestOptions, String> {
        self.request_options.as_deref().map_or_else(
            || Ok(render::standard_request_options()),
            render::request_options,
        )
    }
}

/// What every subcommand that writes a run takes.
#[derive(Args)]
pub struct RunArgs {
    #[command(flatten)]
    pub common: Common,
    /// The run directory.
    #[arg(long, value_name = "DIR")]
    pub output: PathBuf,
    /// The build identity (`{"commit", "dirty_digest"}`) from the evaluation
    /// remote package; the record's build is `null` without it.
    #[arg(long, value_name = "FILE")]
    pub build_identity: Option<PathBuf>,
}

/// The model's host artifacts, opened for rendering as the target serves.
pub fn host(artifacts: &ArtifactsFile) -> Result<HostArtifacts, String> {
    HostArtifacts::open(&PackageOptions {
        target_shards: artifacts.artifacts.target_shards(),
        projector: ProjectorSelection::Disabled,
        draft: None,
    })
    .map_err(|error| error.to_string())
}

/// What every subcommand reads before it runs.
pub struct Inputs {
    pub run: Run,
    pub build: Value,
    pub artifacts: ArtifactsFile,
    pub templates: Vec<TemplateFile>,
    pub request_options: RequestOptions,
}

impl Inputs {
    pub fn read(args: &RunArgs, tool: &'static str) -> Result<Self, String> {
        let build = record::build(args.build_identity.as_deref())?;
        let request_options = args.common.request_options()?;
        let artifacts = ArtifactsFile::read(&args.common.artifacts)?;
        let templates = TemplateFile::read_all(&args.common.templates)?;
        let run = Run::create(args.output.clone(), tool)?;
        Ok(Self {
            run,
            build,
            artifacts,
            templates,
            request_options,
        })
    }

    /// Render every template's lanes for the model's target, as it serves,
    /// templates in order, lanes in order.
    pub fn render(&self) -> Result<Vec<Rendered>, String> {
        let host = host(&self.artifacts)?;
        let renderer = Renderer::new(
            &host,
            &self.artifacts.artifacts.model,
            &self.request_options,
        );
        let mut rendered = Vec::new();
        for file in &self.templates {
            rendered.extend(renderer.render(file)?);
        }
        Ok(rendered)
    }

    /// Write `rendered.jsonl` and return its digest.
    pub fn write_rendered(&self, rendered: &[Rendered]) -> Result<String, String> {
        render::write_rendered(
            &self.run.path("rendered.jsonl"),
            rendered,
            &self.artifacts.artifacts.model,
            &self.artifacts.artifacts.target_sha256,
        )
    }

    pub fn finish(
        self,
        rendered_digest: String,
        devices: Option<Vec<String>>,
        measurements: Map<String, Value>,
        traces: Map<String, Value>,
    ) -> Result<(), String> {
        self.run.finish(RunRecord {
            build: self.build,
            model: self.artifacts.artifacts.model.clone(),
            artifacts: self.artifacts.raw,
            templates: self
                .templates
                .iter()
                .map(|file| record::Template {
                    name: file.name.clone(),
                    context: file.context,
                    path: file.path.clone(),
                    digest: file.digest.clone(),
                })
                .collect(),
            rendered_digest,
            host: record::host(devices),
            measurements,
            traces,
        })
    }
}

/// Carry out the `render` subcommand for `magnitude-harness` and `magnitude-render`.
pub fn run_render(run: &RunArgs) -> Result<(), String> {
    let inputs = Inputs::read(run, "harness.render")?;
    let rendered = inputs.render()?;
    let digest = inputs.write_rendered(&rendered)?;
    inputs.finish(digest, None, Map::new(), Map::new())
}
