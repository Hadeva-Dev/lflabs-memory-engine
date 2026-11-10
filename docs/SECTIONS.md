# Memory sections

Every memory has one section, and sometimes a second one. Sections are labels on the one search index, not separate
stores. They are used in three ways:

- **Ranking:** a soft boost toward the section a session is in (a project folder: `projects`).
- **Scoping:** what each AI app may see. A new app starts at "work only".
- **Browsing:** `lflabs-memory sections`, `lflabs-memory outline [SECTION]` and `OUTLINE.md` in the data folder.

## The sections

| section | what goes there | a "work only" app sees it |
|---|---|---|
| `projects` | work on one project (the memory's project names it) | yes |
| `prefs` | how you want AI tools to work, not tied to one project | yes |
| `reference` | pointers: where things live, URLs, tools and their settings | yes |
| `about` | who you are at work: role, skills, stack, team | yes |
| `people` | about a specific other person | no |
| `personal` | your own life outside work | no |

Health and money (found by a word list, or by the optional sensitivity model) and past chats stay hidden from "work
only" apps, whatever section a memory is in.

## Filing

A memory is filed the moment it is saved, with no model: a fact about you is `personal`; anything else by its project,
else its kind (project: `projects`, feedback: `prefs`, reference: `reference`), else `personal`, so nothing private
reaches a work app by default. With a model
available, the classifier files those default filings again. A filing by the classifier, or one you made, is never
changed. The `filed_by` column says which it was: `rule`, `default` or `model`.

A host assistant can bring its own sections and rules as a profile (`src/lflabs_memory/profiles/`).
