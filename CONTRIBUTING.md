# Contributing to Dictata

Contributions are genuinely welcome — bug reports, ideas, documentation, and
code. Dictata is a small, single-maintainer project, so the bar is not
"be an expert": it is **be deliberate**. A well-scoped fix with a test beats a
large refactor every time.

Please read this before opening a pull request. It is short, and it exists so
that your work does not get closed for a reason you could not have guessed.

## Before you write code

**Open an issue first** for anything beyond a typo or an obvious one-line bug.
Describe what you observed, on which Windows version, with which model and
audio source. This costs you five minutes and can save you an afternoon: some
things that look like bugs are deliberate trade-offs (see the comments in the
code — they usually say why), and some things you might want to add are already
ruled out.

An issue also lets us agree on the approach. A PR that arrives with no prior
discussion and rewrites a subsystem will not be merged, however good it is —
not because of its quality, but because nobody agreed the subsystem should be
rewritten.

## The quality bar

Every PR must satisfy all of the following. There is no negotiation on these,
because each one exists to prevent a class of bug that has already happened.

- **`cargo test` passes.** If you change behaviour, add a test that fails
  before your change and passes after. A behaviour change with no test is not
  reviewable.
- **`cargo clippy` introduces no new warning.** The project is not at zero
  warnings, but it does not go backwards.
- **`cargo fmt` on the code you touched** — and only on the code you touched.
- **No new `unwrap()` / `expect()` in a critical path**: audio callback,
  transcription worker, FFI boundary. A panic in a cpal callback takes the
  process with it.
- **No panic across an FFI boundary**, and no new `unsafe` without a `SAFETY:`
  comment explaining why it holds.
- **No blocking lock in a real-time audio path**, and no mutex held across an
  `.await`.
- **No logging of user data.** Transcripts, window titles, audio, file paths
  under the user's home, tokens. This is not a style preference: the entire
  promise of the application is that nothing leaves the machine, and a log line
  is a leak. Log lengths and identifiers instead — see how `main.rs` logs a
  take (`chars=…`, never the text).
- **Configuration stays backward compatible.** A `config.json` written by an
  older build must keep every setting it had. New fields need
  `#[serde(default = …)]`.
- **Explain the *why* in comments, not the *what*.** The existing comments
  document trade-offs and traps. Match that, and match the surrounding style.

## What gets closed on sight

Not out of hostility — these consume review time that the project does not
have, and none of them make the software better:

- **Bulk or automated pull requests.** If you point a tool at this repository
  and let it open PRs, they will be closed and the account blocked. One
  thoughtful PR is worth more than fifty generated ones, and fifty generated
  ones cost more to triage than they could ever be worth.
- **AI-assisted work that you have not read, run and understood.** Using an
  assistant is fine — that is how a lot of this codebase was written. Shipping
  its output unverified is not. If you cannot explain why your change is
  correct, or you have not run it against a real dictation, do not open the PR.
  Expect to be asked how you tested it.
- **Project-wide reformatting, renaming, or "cleanup"** unrelated to a fix.
- **Dependency bumps with no stated reason.** Say what it fixes.
- **Unrequested features.** Open an issue; scope is discussed before code.
- **Weakening a test, deleting an assertion, or adding `#[allow(…)]`** to make
  something pass. Fix the cause.

## Reporting a bug

Include: Windows version, GPU or CPU build, model, audio source, whether
streaming was on, and what you expected versus what happened.

**Never paste a transcript, an audio file, or a personal path into an issue.**
Anonymise it. If a bug can only be explained with your actual dictation,
describe its shape instead ("a 40-second take in French with two long pauses").

## Working on Linux support

This is the most useful thing anyone could contribute right now, and
[LINUX.md](LINUX.md) lists the blockers in the order they should be tackled.
Start with the CPU build on X11. Please claim a blocker in an issue before
starting — they are large enough that two people duplicating the work would be
a real waste.

## Building and testing

See [README.md](README.md) — in particular the MAX_PATH workaround, which will
bite you on the first Vulkan build.

```powershell
cargo test                              # 64 unit tests, no model needed
cargo clippy --all-targets
cargo build --release --no-default-features   # CPU build, no Vulkan SDK
```

Several parts cannot be covered by unit tests and need a manual check when you
touch them: unplugging a microphone mid-take, cancelling with a long `Esc`,
streaming with real pauses, and launching with an old `config.json`. If your
change is in that territory, say in the PR what you tested by hand.

## Commits and pull requests

- One logical change per PR. If you find a second bug, that is a second PR.
- Present-tense, imperative commit subjects ("Fix the resampler phase across
  slices"), and a body explaining *why* when the subject is not enough.
- Describe in the PR what you changed, why, and how you verified it.

## Licence

Dictata is under the **MIT Licence with the Commons Clause** (see
[LICENSE](LICENSE)). By submitting a contribution you agree that it is
distributed under those terms.

Thanks for being here.
