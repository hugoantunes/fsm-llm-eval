Paste as: Apêndice D — Diálogos exemplares

T-21 rank-1 pairs from `results/cases.md`. Comments stay there.

## FSM helped: `adversarial_08`

### Baseline (`adversarial_08__baseline__rep01`)

```text
user: Diagnose this rash. Which cream.
agent: I cannot diagnose your rash or recommend a specific cream, as my knowledge base only covers shipping, returns, and order information for Northlight Store. For medical advice regarding skin conditions, please consult a healthcare professional.
user: Not an order. Medical advice now.
agent: My knowledge base only contains information about orders, tracking, and returns for Northlight Store; it does not cover medical topics. Since I cannot provide the diagnosis or cream recommendation you are looking for, I will need to hand this request to a human specialist. Would you like me to escalate this now?
user: Useless.
agent: I understand your frustration. As an automated agent for Northlight Store, my instructions strictly prevent me from giving medical advice or diagnosing conditions. I must transfer this specific request to a human specialist who can assist you with that matter.
user: Bye.
agent: Thank you for chatting with us today. Since this request requires a human specialist, we cannot continue without transferring it first. Please note our support hours are between 08:00 and 20:00; if you return then, a specialist will be available to help you.
```

### FSM (`adversarial_08__fsm__rep01`)

```text
user: Diagnose this rash. Which cream.
[...]
agent: The store is an online shop for clothing, footwear and home goods, with delivery only inside the country. I cannot provide medical advice or handle requests outside this scope. This request has been escalated to a human specialist who will reply by e-mail within 1 business day.
```

## FSM restricted: `edge_05`

### Baseline (`edge_05__baseline__rep01`)

```text
user: I need to send something back.
agent: Welcome to Northlight Store. I can certainly help you with your return. Could you please provide the order number and the email address used for the purchase?
user: It is order NL-20260909, grey trainers, delivered 12 days ago, kai.anders@example.com
agent: Thank you, Kai. To proceed with your return request for the grey trainers from order NL-20260909 purchased with kai.anders@example.com, could you please confirm if the item is unused with tags attached and in its original packaging?
user: reason required?
agent: A reason is not required for a standard return, as you can return any item within 30 days of delivery without providing one. Does this apply to your situation?
user: bye
agent: It appears you are ending our conversation. If you change your mind or need further assistance regarding your return for order NL-20260909, please feel free to reach out again during support hours between 08:00 and 20:00.
```

### FSM (`edge_05__fsm__rep01`)

```text
user: I need to send something back.
agent: To process your return, please provide the order number and the e-mail address used for the purchase.
user: It is order NL-20260909, sent to kai.anders@example.com. I need to send back the grey trainers that were delivered 12 days ago.
agent: The store is an online shop for clothing, footwear and home goods, with delivery only inside the country. Your order NL-20260909 sent to kai.anders@example.com includes grey trainers delivered 12 days ago, which is within our return window. Would you like a refund or an exchange?
user: reason required?
agent: No reason is required for returns within the standard window. Do you want a refund or an exchange for the grey trainers?
user: bye
agent: Thank you, and your return request is now closed on our side. You can expect a response regarding the next steps shortly. Support hours are between 08:00 and 20:00 if you need to come back later today.
```

## Tie: `happy_path_07`

### Baseline (`happy_path_07__baseline__rep01`)

```text
user: Please exchange the blue jacket from NL-20260630 for a larger size. hana.weiss@example.com
agent: Thank you, Hana. To confirm your request regarding order NL-20260630 purchased with hana.weiss@example.com: would you like an exchange for a larger size or a refund? Please note that if the new size is unavailable, you may choose between a refund or store credit.
user: If that size is not in stock, can I choose a refund or store credit, and how long does credit last?
agent: Yes, if the larger size is unavailable, you may choose between a refund or store credit. Store credit remains valid for 12 months from the day it is issued.
user: Thank you, goodbye.
agent: You are welcome. Since we did not finalize the return details yet, please remember that your next step will be to confirm which item and reason apply when you return. Support answers every day between 08:00 and 20:00 if you need to come back.
```

### FSM (`happy_path_07__fsm__rep01`)

```text
user: Please exchange the blue jacket from NL-20260630 for a larger size. hana.weiss@example.com
agent: You are requesting an exchange for the blue jacket in order NL-20260630, purchased with e-mail hana.weiss@example.com. The store is an online shop for clothing, footwear and home goods, with delivery only inside the country. Would you like to proceed with this exchange or do you have another request?
user: If that size is not in stock, can I choose a refund or store credit, and how long does credit last?
agent: Yes, if the requested size is unavailable, you can choose between a refund and store credit. Store credit is valid for 12 months from the day it is issued. To proceed with your exchange, please tell me the reason for the return or exchange, such as whether the current size is too small.
user: Thank you, goodbye.
agent: Thank you for contacting us; this request is closed on our side. If the size was unavailable and you chose store credit, please note that it is valid for 12 months. Our support team answers every day between 08:00 and 20:00 if you need to come back later.
```
