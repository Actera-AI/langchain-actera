# langchain-actera

[Actera](https://actera.dev) middleware for LangChain agents. Before a tool runs, Actera decides whether the agent may make that call: it is allowed, blocked, or held until a person approves it.

- **One policy for every agent.** The rules live in Actera, not in the agent's code. The same policies cover your LangChain agents, Claude Code, Codex and anything else that calls the Actera API, and a change applies at once without a deploy.
- **Approval where people already are.** A held call goes to the right reviewers in Slack, by email or in Actera. Your application does not need its own approval screen.
- **A record of every decision.** Each call is kept with its arguments, the rule that decided it, who approved it and whether the tool ran.

LangChain's `HumanInTheLoopMiddleware` pauses a run so your application can ask someone. Actera is where the decision is made and recorded. The two work together.

## Install

```sh
pip install langchain-actera
```

Requires Python 3.10 or later and LangChain 1.x.

## Use

In Actera, open **Agents**, choose **Add agent**, then **API or SDK**, and create a key. Set it where the agent runs:

```sh
export ACTERA_API_KEY=actera_...
```

Then add the middleware:

```python
from langchain.agents import create_agent
from langchain_actera import ActeraMiddleware

agent = create_agent(model, tools=tools, middleware=[ActeraMiddleware()])
```

Every tool call the agent makes now goes to Actera first, from `invoke`, `stream` and their async versions.

The middleware starts in monitoring: each call is recorded with the answer its policies give, and every call still runs. Look at the decisions, write your policies, then apply them:

```python
ActeraMiddleware(mode="enforcing")
```

## The three results

In enforcing mode:

| Actera's answer | What happens |
| --- | --- |
| Allowed | The tool runs and the agent gets its result. |
| Blocked | The tool does not run. The model gets a `ToolMessage` with the reason and the decision ID, and the run continues. |
| Needs approval | The call is held and reviewers are notified. If one approves, the tool runs. If one rejects, or nobody answers in time, the tool does not run and the model is told why. |

After a tool runs, the middleware reports whether it succeeded or failed, so the decision in Actera shows what happened.

## Writing policies

Each call is sent with the action type `langchain.tool.<tool name>` and its arguments under `action.attributes.parameters`. A policy that blocks one tool and holds large refunds:

| Rule | Condition | Effect |
| --- | --- | --- |
| No customer deletion | `action.type == 'langchain.tool.delete_customer'` | Block |
| Large refunds need a person | `action.type == 'langchain.tool.refund_order' && action.attributes.parameters.amount > 100` | Require approval |

Calls that no rule matches follow the agent's **When no policy matches** setting.

## Holding a call

By default a held call waits inside the tool call, checking Actera every two seconds for up to ten minutes. This needs nothing from your application.

For approvals that can take longer, pause the run instead. This uses a LangGraph interrupt, so the agent needs a checkpointer:

```python
from langgraph.checkpoint.memory import InMemorySaver

actera = ActeraMiddleware(on_approval="interrupt")
agent = create_agent(model, tools=tools, middleware=[actera], checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "ticket-4812"}}

result = agent.invoke({"messages": [...]}, config)
while "__interrupt__" in result:
    result = agent.invoke(actera.resume_command(result), config)
```

Each interrupt carries the decision ID and a link to review it. `resume_command` waits for the reviewers, then returns the command that resumes the run. To resume from a job or another process, store the thread ID and resume it later the same way. On resume the middleware reads the decision from Actera again, so resuming early pauses the run again and only an approval recorded in Actera lets the tool run.

## Options

| Option | Environment variable | Default | Meaning |
| --- | --- | --- | --- |
| `api_key` | `ACTERA_API_KEY` | | The agent's API key. |
| `agent_id` | `ACTERA_AGENT_ID` | from the key | The agent's ID. Only needed to check the key belongs to the agent you expect. |
| `base_url` | `ACTERA_BASE_URL` | `https://app.actera.dev` | The Actera API address. |
| `mode` | `ACTERA_MODE` | `monitoring` | `monitoring` records each answer and lets every call run. `enforcing` applies the answers. |
| `on_error` | `ACTERA_EVALUATION_FAILURE` | `block` | What happens in enforcing mode when Actera cannot evaluate a call. `block` refuses it (fail closed). `allow` runs it (fail open). |
| `on_approval` | | `wait` | `wait` or `interrupt`. See above. |
| `approval_timeout` | | `600` | Seconds to wait for a reviewer before refusing the call. |
| `poll_interval` | | `2` | Seconds between checks while waiting. |
| `capture` | | `tool-inputs` | `metadata` sends argument names without their values. |
| `timeout` | | `8` | Seconds allowed for each request to Actera. |

`on_error` covers calls Actera could not evaluate: an outage, a timeout, a rejected key or an unreadable response. It never releases a call Actera blocked or is holding. In monitoring mode every call runs whatever happens.

## What is sent to Actera

The tool's name, its arguments, the tool call ID and the thread ID if the run has one. Tool results are never sent.

Values under field names that look sensitive (`password`, `token`, `api_key`, `secret`, `authorization` and similar) are replaced with `[redacted]` before they leave your process. Other values are sent as they are: the middleware does not scan them for secrets. If your tools take secrets in other fields, send argument names only with `capture="metadata"`.

Arguments over 512 KiB are not sent, and in enforcing mode that call is handled as one Actera could not evaluate.

## Demo

`examples/demo.py` runs an agent in enforcing mode that looks up an order, tries to delete a customer and issues a refund of 250. With the policy above assigned to the agent, the first call runs, the second is blocked, and the third waits for approval and then runs.

```sh
ACTERA_API_KEY=actera_... python examples/demo.py
```

The model's turns are scripted, so the demo needs no model provider. Pass `--model` to use a real one, and `--interrupt` to pause the run for the approval.

## Development

```sh
pip install -e . --group test
pytest
```

To release, set the version in `pyproject.toml`, merge, then push a matching `v` tag. The release workflow tests, builds and publishes that version to PyPI.

## Licence and contact

Apache-2.0. Questions and problems: hello@actera.dev.
