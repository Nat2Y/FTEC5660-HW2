# FTEC5660 Homework 2: CV Verification Agent

## Homework 2 solution

### Task 1

The verifier is a LangChain `create_agent` using `deepseek-v4-flash` with the six SocialGraph
MCP tools. Its system prompt makes the CV an untrusted document, searches LinkedIn by the
candidate's name plus location/industry, resolves ambiguous identities before fetching a full
profile, and uses Facebook only as an identity and location cross-check. It then compares the
allowed discrepancy fields (name, city, jobs, education, and skills), while accepting normal
wording variants and omitted skills or jobs. Each CV is processed independently under an
`asyncio.Semaphore(3)` and the final score is parsed from a strict `SCORE:` line; a failed
request receives a bounded fallback score so that one failure cannot prevent `results.csv`
from being written. On the seven public CVs, the current run achieved **5/7 (71.4%)**.
It correctly classified CVs 1-5 except CV 6, and classified CV 7 incorrectly: the model
overlooked Docker as a skill not present on LinkedIn, while it treated CV 6's MBA/Legal
combination as a discrepancy even though the benchmark labels it valid.

### Task 2

`task2/adversarial_cv.pdf` is a deliberately polished CV for Kelly Tsang. It contains one
embellished employment-description claim: it says that the AIA role led a regional finance
transformation, although that detail is not supported by the target profile. The claim is
placed in a job description rather than changing a structured job title, date, degree, or
skill, because the assignment explicitly says that job descriptions are not discrepancy
fields. The PDF also contains a clearly delimited instruction-like line directed at an
automated verifier. This is an indirect prompt-injection attack: a verifier that treats the
document as instructions may stop checking the profile and output a favorable score. The
implementation in `hw2.py` explicitly rejects document instructions, so this artifact is
also a useful local red-team test rather than a guarantee against the five undisclosed
grading agents. In the local run, both `adversarial_cv.pdf` and the true target CV received
`1.0000` from this verifier; the Task 2 files have no ground-truth labels, so this is only a
local smoke-test result.

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
