"""mem.py connect (connect_apps.py): wiring the AI apps on this Mac to memory, entirely under a fake HOME. A dry run
writes nothing; --yes backs up each file, adds only the lflabs-memory entry (and Codex's prompt hook), keeps every
other byte of TOML and every other key of JSON, writes through symlinks, keeps file modes, never prints a file's
secrets, adds nothing on a rerun, leaves JSON with comments alone, and never writes the Claude app's config while it
runs. No real config is read or written.

  .venv/bin/python -m unittest -q test_connect
"""
import _env  # noqa: F401  (first: the test data folder and user)
import json
import os
import stat
import shutil
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

from lflabs_memory import access
from lflabs_memory import connect_apps
from lflabs_memory import mem

S = connect_apps.SERVER  # our entry's name

# Split so the fixture doesn't trip a secret check on its own source.
FAKE = "sk-" + "test-DO-NOT-PRINT-0123456789abcdef"
LIB = "Library/Application Support"
CODEX_TOML = f'''model = "gpt-5"
notify = ["say", "done"]

[mcp_servers.notes]
command = "npx"
args = ["-y", "notes"]

[mcp_servers.notes.env]
API_KEY = "{FAKE}"

[hooks.state."/x/hooks.json:pre_tool_use:0:0"]
trusted_hash = "abc"
'''
CODEX_HOOKS = {"hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command",
                                                                        "command": "graphify-hint"}]}]}}
CURSOR = {"mcpServers": {"posthog": {"command": "npx", "args": ["-y", "mcp-remote"],
                                     "env": {"POSTHOG_AUTH_HEADER": f"Bearer {FAKE}"}}}}
CLAUDE = {"mcpServers": {}, "preferences": {"menuBarEnabled": True}, "coworkUserFilesPath": "/x"}


class Connect(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp(prefix="memconnect"))
        self.addCleanup(shutil.rmtree, self.home, True)
        self.root = self.home / "lantern" / "memory"
        self.root.mkdir(parents=True)
        launcher = (str(self.root / ".venv/bin/python"), str(self.root / "mem.py"))  # a host's launcher
        for p in (mock.patch.object(mem, "ROOT", self.root), mock.patch.object(mem, "LAUNCHER", launcher),
                  mock.patch.object(connect_apps, "claude_running",
                                                                              lambda: False),
                  mock.patch.object(mem, "KEYLESS_MODE", "on"), mock.patch.object(mem, "TALK_PERMS_MODE", "off"),
                  mock.patch.dict(mem._clients, {"key": None, "data": {}}),
                  # an access change posts a notice: into this fake root, never Robin's feed or a banner (2026-10-24:
                  # a codex widening from test_no_key_in_output_or_the_change_log reached the live notices.jsonl)
                  mock.patch.object(access, "NOTICES", self.root / "notices.jsonl"),
                  mock.patch.object(access, "BANNERS", "off")):
            p.start()
            self.addCleanup(p.stop)
        self.put(".codex/config.toml", CODEX_TOML, 0o600)
        self.put(".codex/hooks.json", json.dumps(CODEX_HOOKS, indent=2) + "\n")
        self.put(".cursor/mcp.json", json.dumps(CURSOR, indent=4), 0o600)
        self.put(f"{LIB}/Claude/claude_desktop_config.json", json.dumps(CLAUDE, indent=2))
        (self.home / ".gemini").mkdir()  # installed, no settings.json yet
        self.put(".claude/settings.json", json.dumps({"hooks": {"UserPromptSubmit": [{"hooks": [
            {"type": "command", "command": "'/x/.venv/bin/python' '/x/mem.py' hook-prompt"}]}]}}))

    def put(self, rel, text, mode=0o644):
        p = self.home / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        os.chmod(p, mode)
        return p

    def run_connect(self, apply=False, only=None):
        lines = []
        code = connect_apps.main(mem, apply=apply, only=only, home=self.home, out=lines.append)
        return code, "\n".join(lines)

    def snapshot(self):
        return {str(p): p.read_bytes() for p in self.home.rglob("*") if p.is_file()}

    def backups(self):
        return sorted(p.name for p in self.home.rglob("*.lflabs-backup-*"))

    def test_dry_run_writes_nothing_and_shows_each_change(self):
        before = self.snapshot()
        code, out = self.run_connect()
        self.assertEqual(code, 0)
        self.assertEqual(self.snapshot(), before)
        self.assertIn("dry run", out)
        for want in ("Claude Code        connected", f"~/.codex/config.toml: add [mcp_servers.{S}]",
                     "~/.codex/hooks.json: add a UserPromptSubmit hook", f'~/.cursor/mcp.json: add mcpServers."{S}"',
                     f'~/.gemini/settings.json: add mcpServers."{S}"', "Windsurf           not installed",
                     '"cursor"', '"--client", "codex"', "LFLABS_MEMORY_CLIENT=codex"):
            self.assertIn(want, out)
        self.assertNotIn(FAKE, out)
        self.assertNotIn("Bearer", out)

    def test_apply_adds_only_ours_and_keeps_everything_else(self):
        code, out = self.run_connect(apply=True)
        self.assertEqual(code, 0, out)
        self.assertNotIn(FAKE, out)
        toml = (self.home / ".codex/config.toml").read_text()
        self.assertTrue(toml.startswith(CODEX_TOML))  # appended, not rewritten
        data = tomllib.loads(toml)
        self.assertEqual(data["mcp_servers"][S]["args"][1:], ["serve", "--client", "codex"])
        self.assertEqual(data["mcp_servers"]["notes"]["env"]["API_KEY"], FAKE)
        hooks = json.loads((self.home / ".codex/hooks.json").read_text())
        self.assertEqual(hooks["hooks"]["PreToolUse"], CODEX_HOOKS["hooks"]["PreToolUse"])
        cmd = hooks["hooks"]["UserPromptSubmit"][0]["hooks"][0]["command"]
        self.assertTrue(cmd.startswith("LFLABS_MEMORY_CLIENT=codex ") and cmd.endswith(" hook-prompt"))
        cursor = json.loads((self.home / ".cursor/mcp.json").read_text())
        self.assertEqual(cursor["mcpServers"]["posthog"], CURSOR["mcpServers"]["posthog"])
        self.assertEqual(cursor["mcpServers"][S]["args"][1:], ["serve", "--client", "cursor"])
        self.assertIn('\n    "mcpServers"', (self.home / ".cursor/mcp.json").read_text())  # its 4-space indent kept
        self.assertEqual(stat.S_IMODE(os.stat(self.home / ".cursor/mcp.json").st_mode), 0o600)
        claude = json.loads((self.home / f"{LIB}/Claude/claude_desktop_config.json").read_text())
        self.assertEqual(claude["preferences"], CLAUDE["preferences"])
        self.assertEqual(claude["mcpServers"][S]["args"][1:], ["serve"])  # known by its reported name
        gemini = json.loads((self.home / ".gemini/settings.json").read_text())
        self.assertEqual(list(gemini), ["mcpServers"])
        # a backup of every file that existed, byte for byte
        self.assertEqual(len(self.backups()), 4)
        b = next(self.home.glob(".codex/config.toml.lflabs-backup-*"))
        self.assertEqual(b.read_text(), CODEX_TOML)

    def test_a_rerun_adds_nothing(self):
        self.run_connect(apply=True)
        after, backups = self.snapshot(), self.backups()
        code, out = self.run_connect(apply=True)
        self.assertEqual(code, 0)
        self.assertEqual(self.snapshot(), after)
        self.assertEqual(self.backups(), backups)
        self.assertEqual(out.count("already has"), 5)

    def test_an_entry_owner_made_is_kept(self):
        mine = {"mcpServers": {"memory": {"command": "/x/python", "args": [str(self.root / "mem.py"), "serve"],
                                          "env": {mem.CLIENT_KEY_ENV: FAKE}}}}
        p = self.put(".cursor/mcp.json", json.dumps(mine))
        code, out = self.run_connect(apply=True, only="cursor")
        self.assertIn(f"already has {S}", out)
        self.assertEqual(json.loads(p.read_text()), mine)
        self.assertNotIn(FAKE, out)

    def test_claude_app_running_is_left_alone(self):
        before = (self.home / f"{LIB}/Claude/claude_desktop_config.json").read_bytes()
        with mock.patch.object(connect_apps, "claude_running", lambda: True):
            code, out = self.run_connect(apply=True)
        self.assertIn("the Claude app is running", out)
        self.assertIn("--only claude-desktop", out)
        self.assertEqual((self.home / f"{LIB}/Claude/claude_desktop_config.json").read_bytes(), before)
        self.assertIn(S, (self.home / ".cursor/mcp.json").read_text())  # the others still done

    def test_json_with_comments_is_never_rewritten(self):
        text = '{\n  // my servers\n  "servers": {}\n}\n'
        p = self.put(f"{LIB}/Code/User/mcp.json", text)
        code, out = self.run_connect(apply=True, only="copilot")
        self.assertEqual(code, 1)
        self.assertIn("not plain JSON", out)
        self.assertIn('"type": "stdio"', out)  # the lines to add by hand
        self.assertEqual(p.read_text(), text)
        self.assertEqual(self.backups(), [])

    def test_vscode_uses_servers(self):
        self.put(f"{LIB}/Code/User/mcp.json", json.dumps({"servers": {"x": {"command": "y"}}}))
        self.run_connect(apply=True, only="copilot")
        d = json.loads((self.home / f"{LIB}/Code/User/mcp.json").read_text())
        self.assertEqual(d["servers"][S]["type"], "stdio")
        self.assertEqual(d["servers"]["x"], {"command": "y"})

    def test_symlinked_config_is_written_through_the_link(self):
        real = self.put("dotfiles/cursor-mcp.json", json.dumps(CURSOR))
        link = self.home / ".cursor/mcp.json"
        link.unlink()
        link.symlink_to(real)
        self.run_connect(apply=True, only="cursor")
        self.assertTrue(link.is_symlink())
        self.assertIn(S, json.loads(real.read_text())["mcpServers"])
        self.assertTrue(list((self.home / "dotfiles").glob("cursor-mcp.json.lflabs-backup-*")))

    def test_broken_toml_is_left_alone(self):
        p = self.put(".codex/config.toml", "model = \n[broken")
        code, out = self.run_connect(apply=True, only="codex")
        self.assertEqual(code, 1)
        self.assertEqual(p.read_text(), "model = \n[broken")
        self.assertIn(f"[mcp_servers.{S}]", out)

    def test_a_failed_check_restores_the_file(self):
        p = self.home / ".codex/config.toml"
        with mock.patch.object(connect_apps, "toml_block", lambda m, c: "[mcp_servers.other]\nx = 1\n"):
            code, out = self.run_connect(apply=True, only="codex")
        self.assertEqual(code, 1)
        self.assertEqual(p.read_text(), CODEX_TOML)


    def test_odd_files_never_stop_the_run(self):
        self.put(".cursor/mcp.json", '{"mcpServers": null}')
        self.put(".codex/hooks.json", '{"hooks": null}')
        self.put(f"{LIB}/Code/User/mcp.json", b"\xff\xfe".decode("latin-1"))
        (self.home / ".gemini/settings.json").mkdir()
        code, out = self.run_connect(apply=True)
        self.assertIn("Claude app", out)  # the last app was still reached
        self.assertIn(f'added mcpServers."{S}"', out)
        self.assertEqual(code, 1)  # the directory at Gemini's path is reported, not hidden

    def test_crlf_files_keep_their_line_endings(self):
        p = self.put(".codex/config.toml", 'model = "x"\r\n[a]\r\nb = 1\r\n')
        self.run_connect(apply=True, only="codex")
        raw = p.read_bytes()
        self.assertTrue(raw.startswith(b'model = "x"\r\n[a]\r\nb = 1\r\n'))
        self.assertNotIn(b"\n", raw.replace(b"\r\n", b""))

    def test_a_memory_memory_entry_for_another_checkout_is_stale(self):
        other = {"mcpServers": {S: {"command": "py", "args": ["/elsewhere/mem.py", "serve"]}}}
        p = self.put(".cursor/mcp.json", json.dumps(other))
        code, out = self.run_connect(apply=True, only="cursor")
        self.assertIn("runs another mem.py", out)
        self.assertEqual(json.loads(p.read_text()), other)

    def test_an_inline_codex_hook_counts(self):
        script = str(self.root / "mem.py")
        self.put(".codex/config.toml", CODEX_TOML + f'\n[[hooks.UserPromptSubmit]]\n[[hooks.UserPromptSubmit.hooks]]\n'
                 f'type = "command"\ncommand = "x {script} hook-prompt"\n')
        code, out = self.run_connect(apply=True, only="codex")
        self.assertIn("already has the prompt hook (in config.toml)", out)
        self.assertNotIn("UserPromptSubmit", (self.home / ".codex/hooks.json").read_text())

    def test_read_only_and_changed_files_are_left_alone(self):
        p = self.put(".cursor/mcp.json", json.dumps(CURSOR), 0o444)
        code, out = self.run_connect(apply=True, only="cursor")
        self.assertIn("read-only", out)
        os.chmod(p, 0o644)
        real_read = connect_apps._read
        calls = []

        def racing(path):  # the app rewrites the file between the plan and the write
            calls.append(path)
            return real_read(path) if len(calls) < 2 else '{"mcpServers": {}}'
        with mock.patch.object(connect_apps, "_read", racing):
            code, out = self.run_connect(apply=True, only="cursor")
        self.assertIn("changed while I was", out)
        self.assertEqual(json.loads(p.read_text()), CURSOR)
        self.assertFalse(list(p.parent.glob("*.lflabs-tmp")))

    def test_only_is_forgiving_and_says_when_nothing_matches(self):
        self.assertIn("~/.cursor/mcp.json", self.run_connect(only="Cursor")[1])
        self.assertIn("Gemini CLI", self.run_connect(only="gemini")[1])
        code, out = self.run_connect(only="nonesuch")
        self.assertEqual(code, 1)
        self.assertIn("no app called", out)

    def test_a_widened_client_gets_a_key_never_printed(self):
        mem.clients_set("cursor", allow=[], deny=[])  # Robin gave Cursor everything
        code, out = self.run_connect()
        self.assertIn("<a new key, written with --yes>", out)
        code, out = self.run_connect(apply=True)
        key = json.loads((self.home / ".cursor/mcp.json").read_text())["mcpServers"][S]["env"][
            mem.CLIENT_KEY_ENV]
        self.assertNotIn(key, out)
        self.assertEqual(mem.resolve_client("cursor", key), ("cursor", True))
        self.assertNotIn("env", json.loads((self.home / ".gemini/settings.json").read_text())["mcpServers"][
            S])  # work only: the name is enough
        before = self.snapshot()
        self.run_connect(apply=True)  # the key is right: nothing rotates
        self.assertEqual(self.snapshot(), before)
        cfg = json.loads((self.home / ".cursor/mcp.json").read_text())
        cfg["mcpServers"][S]["env"][mem.CLIENT_KEY_ENV] = "edited-by-hand"
        self.put(".cursor/mcp.json", json.dumps(cfg))
        code, out = self.run_connect(apply=True, only="cursor")
        self.assertIn("gave its entry a new key", out)
        new = json.loads((self.home / ".cursor/mcp.json").read_text())["mcpServers"][S]["env"][
            mem.CLIENT_KEY_ENV]
        self.assertEqual(mem.resolve_client("cursor", new), ("cursor", True))
        self.assertEqual(mem.resolve_client("cursor", key), ("cursor", False))

    def test_codex_key_goes_into_both_files(self):
        mem.clients_set("codex", allow=[], deny=[])
        self.run_connect(apply=True, only="codex")
        import tomllib
        key = tomllib.loads((self.home / ".codex/config.toml").read_text())["mcp_servers"][S]["env"][
            mem.CLIENT_KEY_ENV]
        cmd = json.loads((self.home / ".codex/hooks.json").read_text())["hooks"]["UserPromptSubmit"][0]["hooks"][0][
            "command"]
        self.assertIn(f"{mem.CLIENT_KEY_ENV}={key} ", cmd)
        self.assertEqual(mem.resolve_client("codex", key), ("codex", True))
        mem._write_clients({"codex": {"deny": []}})  # the hash lost: both files get a new key together
        mem._clients.update(key=None)
        self.run_connect(apply=True, only="codex")
        key2 = tomllib.loads((self.home / ".codex/config.toml").read_text())["mcp_servers"][S]["env"][
            mem.CLIENT_KEY_ENV]
        self.assertNotEqual(key, key2)
        self.assertIn(key2, (self.home / ".codex/hooks.json").read_text())


    def test_a_table_under_the_old_name_counts_and_gets_its_key(self):
        """An entry of ours under the name before 0.1.8 (OLD_SERVER) is connected: no second table is added, and a key
        goes into it under that name."""
        import tomllib
        mem.clients_set("codex", allow=[], deny=[])
        old = connect_apps.OLD_SERVER
        self.put(".codex/config.toml", CODEX_TOML + f'\n[mcp_servers.{old}]\ncommand = "py"\nargs = '
                 f'[{json.dumps(str(self.root / "mem.py"))}, "serve", "--client", "codex"]\n')
        self.run_connect(apply=True, only="codex")
        servers = tomllib.loads((self.home / ".codex/config.toml").read_text())["mcp_servers"]
        self.assertNotIn(S, servers)
        key = servers[old]["env"][mem.CLIENT_KEY_ENV]
        self.assertEqual(mem.resolve_client("codex", key), ("codex", True))
        self.assertEqual(connect_apps.BACKUP, "lflabs-backup")

    def test_files_with_a_key_and_all_backups_are_private(self):
        mem.clients_set("cursor", allow=[], deny=[])
        p = self.put(".cursor/mcp.json", json.dumps(CURSOR), 0o644)
        g = self.put(".gemini/settings.json", "{}", 0o644)  # work only: no key, its mode is kept
        self.run_connect(apply=True)
        self.assertEqual(stat.S_IMODE(os.stat(p).st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(os.stat(g).st_mode), 0o644)
        for b in self.home.rglob("*.lflabs-backup-*"):
            self.assertEqual(stat.S_IMODE(os.stat(b).st_mode) & 0o077, 0, b)
        os.chmod(p, 0o400 | 0o200)  # stricter than 0600 stays as it is
        cfg = json.loads(p.read_text())
        cfg["mcpServers"][S]["env"][mem.CLIENT_KEY_ENV] = "stale"
        p.write_text(json.dumps(cfg))
        self.run_connect(apply=True, only="cursor")
        self.assertEqual(stat.S_IMODE(os.stat(p).st_mode), 0o600)

    def test_no_key_in_output_or_the_change_log(self):
        with mock.patch.object(mem, "TALK_PERMS_MODE", "on"):
            mem.clients_set("codex", allow=[], deny=[])
            code, out = self.run_connect(apply=True)
            for f in (".codex/config.toml", ".codex/hooks.json"):
                os.chmod(self.home / f, 0o600)
            self.put(".codex/hooks.json", "{ not json")  # an error path prints the hook lines to add by hand
            code, out2 = self.run_connect(apply=True, only="codex")
        import tomllib
        key = tomllib.loads((self.home / ".codex/config.toml").read_text())["mcp_servers"][S]["env"][
            mem.CLIENT_KEY_ENV]
        self.assertNotIn(key, out + out2)
        self.assertIn("<a new key, written with --yes>", out2)
        log = self.root / "access-history.jsonl"
        self.assertNotIn(key, log.read_text() if log.exists() else "")
        self.assertNotIn(mem._key_hash(key), log.read_text() if log.exists() else "")

    def test_a_working_key_is_never_rotated_away(self):
        mem.clients_set("codex", allow=[], deny=[])
        key, _ = mem.clients_add("codex", with_key=True)
        cfg = self.put(".codex/config.toml", CODEX_TOML + f'\n[mcp_servers.{S}]\ncommand = "py"\nargs = '
                       f'[{json.dumps(str(self.root / "mem.py"))}, "serve", "--client", "codex"]\n\n'
                       f'[mcp_servers.{S}.env]\n{mem.CLIENT_KEY_ENV} = "{key}"\n', 0o444)  # read-only
        code, out = self.run_connect(apply=True, only="codex")
        self.assertEqual(mem.resolve_client("codex", key), ("codex", True))  # the table's key still works
        cmd = json.loads((self.home / ".codex/hooks.json").read_text())["hooks"]["UserPromptSubmit"][0]["hooks"][0][
            "command"]
        self.assertIn(key, cmd)  # the hook got the same key
        self.assertIn(f"already has {S}", out)
        os.chmod(cfg, 0o644)

    def test_a_backup_that_fails_is_reported_not_raised(self):
        mem.clients_set("codex", allow=[], deny=[])
        locked = self.home / "locked"
        locked.mkdir()
        (locked / "hooks.json").write_text(json.dumps(CODEX_HOOKS))
        (self.home / ".codex/hooks.json").unlink()
        (self.home / ".codex/hooks.json").symlink_to(locked / "hooks.json")
        os.chmod(locked, 0o555)
        try:
            code, out = self.run_connect(apply=True, only="codex")
        finally:
            os.chmod(locked, 0o755)
        self.assertEqual(code, 1)
        self.assertIn("not written (PermissionError)", out)
        import tomllib
        key = tomllib.loads((self.home / ".codex/config.toml").read_text())["mcp_servers"][S]["env"][
            mem.CLIENT_KEY_ENV]
        self.assertEqual(mem.resolve_client("codex", key), ("codex", True))  # what was written is registered

    def test_a_hook_gets_its_key_in_place(self):
        mem.clients_set("codex", allow=[], deny=[])
        script = str(self.root / "mem.py")
        hooks = {"hooks": {"UserPromptSubmit": [{"hooks": [{"type": "command", "command":
                 f"LFLABS_MEMORY_CLIENT=codex FOO=bar py {script} hook-prompt >> /tmp/hook.log 2>&1"}]}]}}
        self.put(".codex/hooks.json", json.dumps(hooks))
        self.run_connect(apply=True, only="codex")
        cmd = json.loads((self.home / ".codex/hooks.json").read_text())["hooks"]["UserPromptSubmit"][0]["hooks"][0][
            "command"]
        self.assertRegex(cmd, r"^LFLABS_MEMORY_CLIENT=codex LFLABS_MEMORY_CLIENT_KEY=\S+ FOO=bar py \S+ hook-prompt "
                              r">> /tmp/hook.log 2>&1$")

    def test_odd_toml_env_is_refused_not_crashed(self):
        mem.clients_set("codex", allow=[], deny=[])
        table = (f'\n[mcp_servers.{S}]\ncommand = "py"\nargs = [{json.dumps(str(self.root / "mem.py"))}, '
                 f'"serve"]\n')
        for env in ('env = "oops"\n', "env = { WHEN = 1979-05-27T07:32:00Z }\n"):
            p = self.put(".codex/config.toml", CODEX_TOML + table + env)
            code, out = self.run_connect(apply=True, only="codex")
            self.assertIn("left alone", out, env)
            self.assertEqual(p.read_text(), CODEX_TOML + table + env)


if __name__ == "__main__":
    unittest.main()
