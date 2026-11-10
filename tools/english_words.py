"""Writes src/lflabs_memory/english-words.txt, the English vocabulary mem.english_screenable counts (run by hand; needs
`pip install wordfreq`, which the engine itself never imports).

The 50,000 most frequent English words, less the common words of the other languages an app might write in: one of a
language's 300 most frequent and at least 10 times as frequent there as in English (de, la, que, sin, und, per, non ...:
they must not count as English; loanwords such as "mode" and "base" stay). Plus TECH: tool names and abbreviations terse
project notes are made of ("pnpm not npm", "deploy fri")."""
import sys
from pathlib import Path

from wordfreq import top_n_list, word_frequency

import trace_scan  # no word the trace scan flags (names) goes in the shipped list

OTHER = ("es", "fr", "de", "it", "pt", "nl", "pl", "tr", "id", "ms", "vi", "sv", "da", "nb", "fi", "cs", "ro", "ca")
TECH = """api apis cli sdk sdks ui ux css scss html js jsx ts tsx py rb go rs json jsonl yaml yml toml csv tsv xml md
env dotenv config configs repo repos monorepo git github gitlab ci cd pr prs mr npm pnpm npx yarn bun deno node nodejs
pip pipx uv conda poetry venv cargo rustc rustup gem bundler brew homebrew make cmake gradle maven docker dockerfile
compose k8s kubectl helm terraform tf aws gcp azure s3 ec2 iam vpc lambda cdn dns ssl tls https http url urls uri ssh
scp rsync cron crontab systemd launchd vpn sso oauth jwt saml ldap db dbs sql sqlite postgres postgresql psql mysql
redis mongo mongodb kafka rabbitmq grpc graphql rest webhook webhooks websocket ws localhost tcp udp ip ipv4 ipv6 port
ports vite webpack esbuild rollup babel eslint prettier ruff black mypy pytest jest vitest mocha playwright cypress
storybook tailwind react nextjs nuxt vue svelte angular django flask fastapi rails laravel spring dotnet csharp kotlin
swift swiftui xcode ios ipados macos android linux ubuntu debian alpine nginx caddy apache vercel netlify fly heroku
render cloudflare supabase firebase stripe sentry datadog grafana prometheus slack jira linear notion figma zod prisma
drizzle orm migrations staging prod dev qa uat repl stdout stderr stdin tty shell bash zsh fish sudo chmod chown regex
async await sync todo fixme wip changelog readme semver lts rc beta alpha hotfix bugfix refactor lint linter
linting typecheck codegen backend frontend fullstack devops infra sre oncall runbook postmortem rollout rollback
canary deploys deployed redeploy cpu gpu ram ssd ms kb mb gb tb utc pst est cet am pm mon tue tues wed thu thur thurs
fri sat sun jan feb mar apr jun jul aug sep sept oct nov dec eng mgr pm asap fyi eta tbd tba btw imo ok okay vs etc
misc dept ext approx min max avg qty workspace workspaces""".split()


# English too, though common elsewhere: they count as English words, never as function words
SHARED = """no son come men door war care ten plus hat mayor premier till dock barn bent holder suite genre coup fare inn
toe tale dies solo era chose gut ham jam dig sue bare mere folk gang tag cap pot pan cop ban saint met net pro hi"""


def build(n=50000):
    common = {lang: set(top_n_list(lang, 300)) for lang in OTHER}
    words = []
    for w in top_n_list("en", n):
        if not w.isalpha():
            continue
        en = word_frequency(w, "en")
        if any(w in common[lang] and word_frequency(w, lang) >= 10 * en for lang in OTHER):
            continue
        words.append(w)
    return sorted(w for w in set(words) | set(TECH) | set(SHARED.split()) if not trace_scan.hits("-", w))


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "src" / "lflabs_memory" / "english-words.txt"
    words = build()
    out.write_text("\n".join(words) + "\n")
    print(f"{len(words)} words -> {out}", file=sys.stderr)
