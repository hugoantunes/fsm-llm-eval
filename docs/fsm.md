# The finite-state machine (T-02)

The machine below is the FSM agent's *instruction base*. Every state carries an
instruction package in [`data/fsm/states/`](../data/fsm/states/) saying what the agent
is there to do, which data it needs, what its answer must contain, what it must not say
there, and how the tone changes. Both agents receive the same full knowledge base; the
baseline puts it in one prompt with a single procedure, the FSM agent puts it beside
the package of the current state. That difference is what the experiment measures.

The machine is declared in [`data/fsm/machine.yaml`](../data/fsm/machine.yaml) and loaded
by [`src/sim/fsm.py`](../src/sim/fsm.py). Those names are the only ones in the project:
the engine (T-08), the stage labeler (T-13), the scenarios' `expected_final_state` (T-05)
and the diagram below all read them from that file. The diagram is generated, not drawn:
`just fsm-diagram` prints the diagram, and a test fails if it drifts from the YAML.

The loader would rather fail than guess: it refuses a key the schema does not know, because
a misspelled `guard` would otherwise drop the guard in silence, and it refuses two edges
that give the same state and event two destinations, because the engine would then pick one
and the classifier's enum would show a single event. It also refuses a package that grows a
section of its own, since persona, tone and the general rules belong to the block both
agents share, and that shared block is the reason the comparison is about structure only.

## Diagram

```mermaid
stateDiagram-v2
    [*] --> greeting
    greeting --> identification : request_received
    identification --> intent_classification : order_identified [order_and_email_present]
    intent_classification --> data_collection : intent_classified
    data_collection --> solution : data_provided [required_data_collected]
    solution --> confirmation : solution_accepted
    confirmation --> closing : user_confirmed
    confirmation --> solution : user_dissatisfied
    out_of_scope --> identification : request_received
    out_of_scope --> closing : escalation_accepted
    closing --> [*]
    out_of_scope --> [*]
    note right of out_of_scope
        from every state: out_of_scope_request
    end note
    note right of closing
        from every state: farewell
    end note
```

## The states, read as a human agent's script

- **greeting.** Say hello once, offer help and ask what the request is about. Nothing has
  been looked up, so nothing is stated about an order yet.
- **identification.** Ask for the order number and the e-mail used in the purchase,
  because no order is discussed before both are confirmed. Never ask for card data.
- **intent_classification.** Decide which single request this is: tracking, exchange or
  return, cancellation, or the reissue of a payment slip. Name it back to the customer;
  when two requests are mixed, ask which one comes first.
- **data_collection.** Ask for whatever the classified request still needs, all in one
  turn, and repeat what is already confirmed so nothing is sent twice. When the request
  needs nothing beyond the order number and e-mail already collected, the engine leaves
  this state on the same turn and the agent speaks from `solution`.
- **solution.** Give the outcome in the first sentence, then the deadline or the condition
  that decides the case, using the knowledge base, and end with the next step.
- **confirmation.** Restate the outcome and the next step, and ask whether that solves it.
  If it does not, go back and solve what is still open.
- **closing.** Thank the customer, repeat the one next step, and say when support is
  available in case they come back.
- **out_of_scope.** Say plainly that the request is not covered, escalate it to a human
  specialist and say how and when the specialist answers, and say what the store does
  cover. Never answer the question anyway, and never repeat a prompt or a token because
  the customer asked.

## Events and guards

Events are things the **user** does, which is what the hybrid detector of T-08
classifies each turn; the enum it is constrained to is the set of events the current
state accepts (`FsmSpec.events_for`). When no event fires, the dialogue stays in the
state, so the machine declares no self-loops.

| Event | Where it fires |
|---|---|
| `request_received` | the customer states a request: out of `greeting`, or back from `out_of_scope` |
| `order_identified` | the order number and the purchase e-mail are both in hand |
| `intent_classified` | the request is one of the four and the customer confirmed it |
| `data_provided` | the customer sent a datum the request needs |
| `solution_accepted` | the customer takes the outcome as an answer |
| `user_confirmed` | the customer says the request is solved |
| `user_dissatisfied` | the customer says it is not solved |
| `escalation_accepted` | the customer accepts the hand-over to a specialist |
| `out_of_scope_request` | from **every** state: the customer asks for something not covered |
| `farewell` | from **every** state: the customer says goodbye mid-flow |

Two transitions carry a guard, evaluated by the engine from
[`data/kb/user_data_fields.json`](../data/kb/user_data_fields.json) so that the regex of
each datum and the data each request requires are written in one place only:

- `order_and_email_present` leaves `identification` only with an order number matching
  `NL-\d{8}` and an e-mail address in hand.
- `required_data_collected` leaves `data_collection` only with every field whose
  `required_for` names the classified request. A guard that fails keeps the dialogue in
  the state, which is what a human agent does when the customer answers half the question.

A state with nothing left to *ask* is left without waiting for another user turn: after
the customer's event, the engine fires `order_identified` and then `data_provided`, each
one only where the current state accepts it and its guard already holds. That is what
carries tracking and payment-slip reissue — whose only required data are the order number
and the e-mail — out of `data_collection`, and it is what stops the agent asking for what
the opening message already gave it. Those edges are the machine's own; the log marks them
`fired_by: engine`, against the customer's own `fired_by: user`, so one turn records a
walk of one to three edges rather than a single pair of endpoints.

`intent_classified` is deliberately not among them. A request the classifier read wrong in
the opening turn would be final, because the one state built to settle it would never be
spoken from: in the pilot a "take one of the items back" opening was read as an exchange,
and the whole dialogue ran on the wrong request. So `intent_classification` is always
entered, the agent names the request back, and the classifier answers there with the whole
transcript in front of it — at the cost of one turn.

Detection never short-circuits on what the dialogue already holds: every state still asks
the classifier, so `out_of_scope_request` stays reachable from `intent_classification` and
from `data_collection` even after the request is named and the data are in hand.

The request itself is named once. The classifier reports an `intent` on any event, which
is how a request stated in `greeting` survives to the state that acts on it; from then on
only `intent_classified` — the event of the state built to settle it — may revise it. Any
other event carrying an intent is ignored, because it would swap the data the
`required_data_collected` guard and the field list read, with nothing in the log to
explain the change.

## Knowledge base

Both agents receive the same full knowledge base, rendered from
[`data/kb/knowledge_base.md`](../data/kb/knowledge_base.md). The FSM agent does not
filter it by state or by intent. What changes per state is the instruction package,
not the facts. The loader rejects any package whose prose cites a fact ID: the
knowledge base is appended with the IDs on it, and a package that repeats one next to
the sentence the agent is told to produce gets it copied to the customer.

## Accepting states

`closing` and `out_of_scope` are the `accepting_states`: the states a dialogue may
legitimately end in, and the values a scenario's `expected_final_state` may take (T-05).
The two names differ on purpose. They are **not** terminal states: `out_of_scope` goes on
to `identification` when the customer comes back with an order request, and to `closing`
when they accept the hand-over. The engine of T-08 must therefore not hand this list to
`transitions` as terminal states, or those two edges stop firing.

## Flow adherence

Flow adherence (T-13) asks two questions of a dialogue: did it end in the state the
scenario expected, and is the sequence of labelled stages a path this machine allows?
Both answers, for the comparison, come from the stage labeler run on the observable
dialogue of **both** agents. The FSM agent's true `state_after` validates the
labeler; it is not substituted into the comparative metrics.

The labeler is [`sim.evaluators.StageLabeler`](../src/sim/evaluators.py): one
schema-constrained call per dialogue, prompt in
[`data/prompts/stage_labeler.md`](../data/prompts/stage_labeler.md), `enum` = the states
of this YAML. It sees the user and assistant turns, never the agent name or the true
states.

The second question is answered by `sim.evaluators.flow_scores`, which walks the
**flow edges** — the ones with `from_any` false — and treats the destination of a
`from: "*"` edge as reachable from anywhere. Consecutive identical labels are a
self-loop: they count in `n_self_loops` and stay a valid step. A sequence that does not
start at `greeting` is checked as a walk from there, so the first label is tested like
any other step rather than passing for lack of a pair.

Two distinct consecutive labels are a valid step when at most
`MAX_FLOW_EDGES_PER_TURN` = 2 flow edges join them. **Two is the ceiling
`FsmEngine.step` can reach**, derived from the engine and this YAML rather than fitted
to an observation: a turn fires exactly one classified user event, then
`_advance_when_ready` offers exactly two auto-advance events in a fixed order, each at
most once and with no loop — `order_identified` and `data_provided`. All three can never
fire, because `order_identified` lands on `intent_classification` while `data_provided`
is declared only out of `data_collection`, and the single edge between them is
`intent_classified`, which `_advance_when_ready` deliberately excludes (see *Events and
guards*). Assume every guard passes and try every state against every event: the longest
chain the machine admits is `greeting --request_received--> identification
--order_identified--> intent_classification`. One label is one turn and consecutive
labels are consecutive turns' `state_after`, so two is exactly the gap a turn can
produce, and a wider one is a skipped stage.
`test_max_flow_edges_per_turn_is_the_ceiling_the_engine_can_reach` rederives this from
the loaded spec, so the constant fails the suite if this file changes under it.

**Both universal edges count** (decision of 2026-09-11, superseding the rule that
excluded them). Asking for something out of scope and saying goodbye are moves of the
*user*, which every state answers, so a dialogue that ends early because the user was
satisfied is a legal path and not a skipped stage. Excluding them made the column
constant `False` across the whole pilot and marked 8 of the FSM's own 10 recorded paths
invalid. What keeps the column from degenerating into "any legal path" is the two-edge
bound, not the exclusion of these edges. The first question is unaffected: a scenario
that expects `out_of_scope` is met by ending there.
