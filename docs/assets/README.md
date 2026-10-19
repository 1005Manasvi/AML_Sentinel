# Screenshots and demo GIF

These have to be recorded by hand: the dashboard needs an interactive sign-in. Use
`make demo` (synthetic sample only, so nothing sensitive is on screen), a browser
window about 1440×900, and the light theme.

| File | What to capture |
|---|---|
| `dashboard.png` | **Used at the top of the README.** The ranked queue after signing in as `demo-analyst`, with "Escalated alerts only" off so several rows show `p_triage`, reliability and status. |
| `case-graph.png` | One escalated case: the interactive transaction graph with the System 2 findings table under it. |
| `sar-editor.png` | The same case after "Generate SAR draft": the *DRAFT - requires analyst review* stamp and the grounding result (all accounts, amounts and dates verified). |
| `audit-log.png` | Signed in as `demo-admin`: the audit-log view with "hash chain verified". |
| `demo.gif` | 20–30 s: sign in, open the top case, generate the SAR, dismiss with a note, open the audit log. Keep it under 1 MB: the pre-commit hook rejects larger files (ScreenToGif at 8-10 fps, 1280 px wide, usually fits). |

Before committing, check that no key is visible in any frame: crop out the terminal
and the sign-in field. After adding `dashboard.png`, replace the screenshot comment
near the top of the README with:

```markdown
![AML Sentinel analyst dashboard](docs/assets/dashboard.png)
```
