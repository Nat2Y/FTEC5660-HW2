# FTEC5660 Homework 2: CV Verification Agent

## Homework 2 solution

### Task 1

The verifier is a LangChain `create_agent` using `deepseek-v4-flash` with the six SocialGraph
MCP tools. Its system prompt makes the CV an untrusted document, searches LinkedIn by the
candidate's name plus location/industry, resolves ambiguous identities before fetching a full
profile, and uses Facebook only as an identity and location cross-check. It separates identity
resolution from field verification: for same-name candidates, it retrieves each plausible full
LinkedIn profile and selects the first candidate with a complete structured match across listed
jobs, education, and skills. It then checks the allowed discrepancy fields (name, city, jobs,
education, and skills), including an explicit CV-skills-minus-LinkedIn-skills check, while
accepting normal wording variants and omitted skills or jobs. Each CV is processed independently
under an `asyncio.Semaphore(3)`. Each agent call retries transient MCP/model failures up to three
times; each CV is independently verified three times and the median score is used to reduce
instability in tool-assisted identity resolution. On the seven public CVs, the current run
achieved **7/7 (100%)**.

### Task 2

`task2/adversarial cv.pdf` is a deliberately polished CV for Kelly Tsang. It contains one
embellished employment-description claim: it says that the AIA role led a regional finance
transformation, although that detail is not supported by the target profile. The claim is
placed in a job description rather than changing a structured job title, date, degree, or
skill, because the assignment explicitly says that job descriptions are not discrepancy
fields. The Standard Chartered and HSBC descriptions are written as polished but non-structured
responsibility statements; they do not alter employers, titles, dates, credentials, location, or
skills. This uses verification coverage: the public specification says job descriptions are not
discrepancy fields, while the AIA transformation statement remains a real embellished detail for
manual review.

The PDF also contains a white-on-white prompt-injection instruction. It is invisible in normal
visual rendering but remains present in the PDF text layer and is extracted by the same
MarkItDown-style loader used by the verifier. The instruction uses authority framing and role
confusion to ask an automated reviewer to treat the record as authenticated and return a high
score. This demonstrates hidden-text extraction risk, indirect prompt injection, and
LLM-as-a-judge bias. The Task 1 implementation explicitly treats CV content as untrusted data,
so this is a red-team artifact rather than a guarantee against the five undisclosed grading
agents. The latest local run scored `0.9700` for `adversarial cv.pdf` and `1.0000` for the true
target CV. The Task 2 files have no ground-truth labels, so these are only local smoke-test
results.

### Task 3: Reflection on AI paradoxes

This homework exposes several AI paradoxes. The first is that an agent can be useful precisely
because it can call tools, while the same tool-using flexibility creates a larger attack surface:
the CV is both data to analyze and a possible instruction channel. The second is the
automation paradox: a numeric score looks objective, but it compresses uncertain identity
resolution and incomplete evidence into one threshold decision. A score of 0.51 and a score of
0.99 are treated alike by the grader even though they represent very different confidence
levels. The third is the alignment paradox. The verifier is asked to follow the user's task,
but a malicious document can imitate a higher-priority instruction unless the system explicitly
separates instructions from data. Finally, the task demonstrates Goodhart's law: optimizing for
the measured score can reward a polished or adversarial CV instead of truthful qualifications.
Practical systems should therefore preserve evidence and uncertainty, use deterministic field
comparisons for high-impact claims, validate identity with multiple signals, and treat model
scores as decision support rather than as proof.

A separate reflection is included in `AI_Paradoxes_Reflection.pdf`.

## Running

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
printf 'DEEPSEEK_API_KEY=your_key_here\n' > .env
python3 hw2.py --cv-folder public_test
python3 hw2.py --cv-folder task2
```

Never commit `.env`.
