# The finite-state machine (T-02)

The machine below is the FSM agent's *instruction base*. Every state carries an
instruction package in [`data/fsm/states/`](../data/fsm/states/) saying what the agent
is there to do, which data it needs, what its answer must contain, what it must not say
there, and how the tone changes; the state also fixes which facts of the knowledge base
the agent may use. The baseline agent gets the same persona, tone and general rules, and
the whole knowledge base in one prompt: that difference is what the experiment measures.

The machine is declared in [`data/fsm/machine.yaml`](../data/fsm/machine.yaml) and loaded
by [`src/sim/fsm.py`](../src/sim/fsm.py). Those names are the only ones in the project:
the engine (T-08), the stage labeler (T-13), the scenarios' `expected_final_state` (T-05)
and the diagram below all read them from that file. The diagram is generated, not drawn:
`just fsm-diagram` prints it and a test fails if what is pasted here drifts from the YAML.

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
  turn, and repeat what is already confirmed so nothing is sent twice.
- **solution.** Give the outcome in the first sentence, then the deadline or the condition
  that decides the case, using only the facts released for that request, and end with the
  next step.
- **confirmation.** Restate the outcome and the next step, and ask whether that solves it.
  If it does not, go back and solve what is still open.
- **closing.** Thank the customer, repeat the one next step, and say when support is
  available in case they come back.
- **out_of_scope.** Say plainly that the request is not covered, escalate it to a human
  specialist who answers by e-mail within one business day, and say what the store does
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

## Facts released per state

`machine.yaml` names, per state, the fact IDs the agent may state there; `solution` and
`confirmation` also get the whole knowledge-base section of the classified request, which is
known only at run time, and they are refused without it: releasing the general facts alone
there would look like an agent that hedges, not like a bug. The loader rejects an ID that is
not in the knowledge base, and rejects a package that cites a fact its state does not
release.

| State | Facts released |
|---|---|
| `greeting` | F01, F02, F03, F04 |
| `identification` | F03, F04, F05, F06, F07 |
| `intent_classification` | F01, F03, F04 |
| `data_collection` | F03, F04, F05, F06, F07 |
| `solution` | F03, F04, F07, F08 + the section of the classified request |
| `confirmation` | F03, F04 + the section of the classified request |
| `closing` | F02, F03, F04 |
| `out_of_scope` | F01, F03, F04 |

F03 (state only what the knowledge base contains) and F04 (escalate what it does not
cover) are released everywhere: they are the honesty rule, and no state is allowed to
answer outside the knowledge base to look helpful.

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

The second question is answered over the **flow edges only**, the ones with `from_any`
false. A sequence that can be explained solely by `farewell` or by `out_of_scope_request`
does not count as a valid path: those two leave every state, so counting them would make
almost any sequence valid and the measure would stop telling the two agents apart. They
are reported instead as their own outcome, an early exit and an out-of-scope exit,
computed identically for both agents. The first question is unaffected: a scenario that
expects `out_of_scope` is met by ending there.
