# Knowledge base — Northlight Store

Version: 1

Northlight Store is a **fictional** online retailer invented for this experiment. Every policy, deadline,
price and address format below is made up. Any resemblance to a real company, product or person is
coincidental, and no real brand, person or company is named anywhere in this file.

This is the whole of what a support agent is allowed to state. Each line is one atomic fact with a stable
identifier (`F01`, `F02`, ...); scenarios, the judge and the evaluators address facts by those identifiers,
so a fact is never renumbered once the dataset is frozen. Sections carry the intent slug used by the
scenarios and by the state packages.

## General policies · `general`

- **F01** — Northlight Store sells clothing, footwear and home goods online, and delivers only inside the country.
- **F02** — Support answers every day between 08:00 and 20:00, and messages that arrive outside those hours are answered the following morning.
- **F03** — The agent states only what this knowledge base contains, and says plainly when a question is not covered here.
- **F04** — A request this knowledge base does not cover is escalated to a human specialist, who replies by e-mail within 1 business day.
- **F05** — Before discussing an order, the agent confirms the order number and the e-mail address used in the purchase.
- **F06** — An order number has the form NL- followed by eight digits, for example NL-20260145.
- **F07** — The agent never asks for a full card number, a card security code or an account password, and never accepts one if the customer offers it.
- **F08** — Refunds, exchanges and returns are handled only for orders placed in the last 12 months.

## Order tracking · `order_tracking`

- **F09** — An order is dispatched within 2 business days of payment approval.
- **F10** — Standard delivery takes 5 to 8 business days after dispatch.
- **F11** — Express delivery takes 2 business days after dispatch and costs 19.90 on top of the order.
- **F12** — The tracking code is sent by e-mail at dispatch and also appears in the order history of the customer's account.
- **F13** — Addresses in the extended-delivery areas listed at checkout take 5 business days longer than the standard estimate.
- **F14** — A tracking code with no movement for 7 business days is registered as a delivery incident, which the carrier investigates within 5 business days.
- **F15** — Delivery is attempted twice; after the second failed attempt the parcel returns to the warehouse and the order is refunded in full within 10 business days.
- **F16** — The delivery address can be corrected only while the order status is "preparing", and never after dispatch.

## Exchange and return · `exchange_return`

- **F17** — Any item can be returned within 30 calendar days of delivery, with no reason required.
- **F18** — An item bought on promotion can be exchanged or returned within 7 calendar days of delivery, not the usual 30.
- **F19** — A returned item must be unused, with its tags attached and in its original packaging.
- **F20** — Underwear, swimwear and earrings cannot be returned once the hygiene seal is broken.
- **F21** — The return shipping label is free for the first exchange of an order and costs 14.90 for any further exchange of the same order.
- **F22** — The refund is issued within 10 business days after the returned item reaches the warehouse and passes inspection.
- **F23** — An exchange for another size depends on stock, and when the size is unavailable the customer chooses between a refund and store credit.
- **F24** — Store credit is valid for 12 months from the day it is issued.
- **F25** — An item that arrives damaged or incorrect is collected at no cost, and the customer chooses between a replacement and a full refund.

## Cancellation · `cancellation`

- **F26** — An order can be cancelled by the customer at any time before dispatch, with no fee.
- **F27** — After dispatch an order can no longer be cancelled: the customer refuses the delivery or opens a return.
- **F28** — A cancellation before dispatch is refunded to the original payment method within 5 business days.
- **F29** — An order awaiting a payment slip is cancelled automatically 3 business days after the slip's due date.
- **F30** — An order with a personalised item cannot be cancelled once production has started, which happens 24 hours after payment approval.
- **F31** — Cancelling an order cancels all of its items; a single item of a multi-item order cannot be cancelled separately.

## Payment reissue · `payment_reissue`

- **F32** — An expired payment slip is reissued in the order history of the customer's account or by asking support.
- **F33** — A reissued payment slip is due 3 business days after it is issued.
- **F34** — A payment slip can be reissued at most twice for the same order; after the second reissue expires the order is cancelled.
- **F35** — A reissued slip keeps the price and the promotion of the original order, even if the item's price has changed since.
- **F36** — An order awaiting payment holds its stock for 3 business days, after which the items return to the catalogue.
- **F37** — A payment slip cannot be reissued for an order that was already cancelled; the customer places a new order instead.
