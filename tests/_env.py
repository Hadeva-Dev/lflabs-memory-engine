"""Every test module imports this first: one temp data folder for the whole run, whose config.toml names the test
user Robin (a plain install, the generic profile). No test touches a real data folder."""
import os
import tempfile
from pathlib import Path

if not os.environ.get("LFLABS_MEMORY_HOME"):
    home = Path(tempfile.mkdtemp(prefix="lflabs-memory-test-"))
    (home / "config.toml").write_text('version = 1\nprofile = "generic"\nowner_name = "Robin"\n')
    os.environ["LFLABS_MEMORY_HOME"] = str(home)
os.environ["LFLABS_MEMORY_UNDER_TEST"] = "1"
os.environ.setdefault("LFLABS_MEMORY_NOTIFY", "off")
# The suite may itself run in an AI app's shell (Claude Code's Bash tool sets CLAUDECODE): commands it runs would then
# count as the app's (mem.run_by_agent). A test that wants that sets the marker itself.
for _k in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "AI_AGENT", "CODEX_SANDBOX", "CODEX_SANDBOX_NETWORK_DISABLED",
           "CODEX_THREAD_ID", "CODEX_CI", "CURSOR_AGENT", "CURSOR_TRACE_ID"):
    os.environ.pop(_k, None)  # no banner on the screen, no notice feed, unless a test asks
