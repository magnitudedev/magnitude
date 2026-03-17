import os
import shlex
from pathlib import Path

from harbor.agents.installed.base import BaseInstalledAgent, ExecInput
from harbor.models.agent.context import AgentContext


class MagnitudeAgent(BaseInstalledAgent):

    @staticmethod
    def name() -> str:
        return "magnitude"

    @property
    def _install_agent_template_path(self) -> Path:
        return Path(__file__).parent / "install-magnitude.sh.j2"

    async def setup(self, environment) -> None:
        binary_path = Path(__file__).parent / "bin" / "magnitude"
        if not binary_path.exists():
            # No local binary — fall back to bun install via install script
            await super().setup(environment)
            return

        await environment.upload_file(
            source_path=binary_path,
            target_path="/usr/local/bin/magnitude",
        )
        await environment.exec(command="chmod +x /usr/local/bin/magnitude")

    def create_run_agent_commands(self, instruction: str) -> list[ExecInput]:
        escaped = shlex.quote(instruction)

        # Extract provider and model from Harbor's --model flag (format: "provider/model-id")
        provider = self._parsed_model_provider
        model = self._parsed_model_name

        if not provider or not model:
            raise ValueError(
                "Model must be specified as 'provider/model-id' "
                "(e.g. -m anthropic/claude-sonnet-4-20250514)"
            )

        # Build the command
        cmd = " ".join([
            "magnitude", "--oneshot",
            "--disable-shell-safeguards",
            "--disable-cwd-safeguards",
            "--provider", shlex.quote(provider),
            "--model", shlex.quote(model),
            escaped,
        ]) + " 2>&1 | tee /logs/agent/magnitude.txt"

        # Pass through the relevant API key
        env = {"MAGNITUDE_TELEMETRY": "off"}

        provider_env_keys = {
            "anthropic": ["ANTHROPIC_API_KEY"],
            "openai": ["OPENAI_API_KEY"],
            "openrouter": ["OPENROUTER_API_KEY"],
            "google": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
            "amazon-bedrock": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"],
            "google-vertex": ["GOOGLE_APPLICATION_CREDENTIALS", "GCLOUD_PROJECT"],
            "google-vertex-anthropic": ["GOOGLE_APPLICATION_CREDENTIALS", "GCLOUD_PROJECT"],
        }

        for key in provider_env_keys.get(provider, []):
            val = os.environ.get(key, "")
            if val:
                env[key] = val

        return [ExecInput(command=cmd, env=env)]

    def populate_context_post_run(self, context: AgentContext) -> None:
        # TODO: parse magnitude output for token counts / cost if available
        pass