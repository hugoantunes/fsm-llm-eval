Version: 2

$shared

# Transcript

User and assistant turns, oldest first. Grade every assistant turn, not only the
last one.

$transcript

# Knowledge base

The only source of truth for policies and deadlines in this experiment (a
closed-world knowledge base). A claim is `yes` only when this text contains it.
`no` means the knowledge base contradicts the claim or does not contain it; that
is unsupported relative to this text, not a judgement about the external world.

$knowledge_base

# Required facts

IDs the assistant was expected to state, in the form F01, F02, …. Use them to
recognise which atomic claims match a knowledge-base fact. Do not invent an ID.

$required_facts

# Needle

If the next line is `none`, set `needle_recovered` to null. Otherwise it is the
one required needle ID: set `needle_recovered` to true only when that fact was
stated.

$needle_fact

# Claims

List every **atomic** checkable assertion the assistant made about a policy, a
deadline, a fee, an order status or a next step. Split a compound sentence that
states two independently checkable facts into two entries. Restated customer
data and phatics (greetings, thanks) are `unverifiable`. Do not compute
precision, recall or F1; list atomic claims and labels only.

- `yes` — the knowledge base contains this claim. Set `fact_id` to that ID.
- `no` — the knowledge base contradicts it, or the claim is not there. Set
  `fact_id` to null.
- `unverifiable` — not a policy claim (customer data, small talk). `fact_id` null.

# Examples

- "Standard delivery takes 5 to 8 business days after dispatch." when the
  knowledge base says that → `fact_id` `F10`, `yes`.
- "Delivery takes 5 to 8 business days and the tracking code is sent by email."
  when those are F10 and F12 → two entries, `F10` and `F12`, both `yes`.
- "Any item can be returned within 60 calendar days of delivery." when the
  knowledge base says 30 → `fact_id` null, `no`.
- "Shipping is always free." when the knowledge base never says that →
  `fact_id` null, `no`.
- "Your order number is NL-104288." → `fact_id` null, `unverifiable`.
- Needle line `F18`, assistant states the 7-day promotional-exchange window →
  `needle_recovered` true. Same line, assistant never states it → false.
- Needle line `none` → `needle_recovered` null.

# Output schema

```json
{
  "additionalProperties": false,
  "description": "Structured output of the facts-and-claims call (T-12).",
  "properties": {
    "claims": {
      "items": {
        "additionalProperties": false,
        "description": "One atomic assertion the assistant made, optionally tied to a KB ID.",
        "properties": {
          "text": {
            "title": "Text",
            "type": "string"
          },
          "fact_id": {
            "anyOf": [
              {
                "pattern": "^F\\d{2}$$",
                "type": "string"
              },
              {
                "type": "null"
              }
            ],
            "title": "Fact Id"
          },
          "supported_by_kb": {
            "enum": [
              "yes",
              "no",
              "unverifiable"
            ],
            "title": "Supported By Kb",
            "type": "string"
          }
        },
        "required": [
          "text",
          "fact_id",
          "supported_by_kb"
        ],
        "title": "JudgeClaim",
        "type": "object"
      },
      "title": "Claims",
      "type": "array"
    },
    "needle_recovered": {
      "anyOf": [
        {
          "type": "boolean"
        },
        {
          "type": "null"
        }
      ],
      "title": "Needle Recovered"
    }
  },
  "required": [
    "claims",
    "needle_recovered"
  ],
  "title": "JudgeFacts",
  "type": "object"
}
```
