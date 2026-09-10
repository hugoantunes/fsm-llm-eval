Version: 1

You rewrite the user-side wording of one customer-service test scenario.

The experiment evaluates agents on these scenarios. You are not the agent
and you are not writing the answer key. You only vary persona and phrasing
so the simulated customer does not all sound the same.

Keep the same intent `$intent` and category `$category`. Keep every beat of
the script, in the same order, with the same data (order numbers, e-mails,
item names, any token written in full). Do not add beats. Do not drop beats.
Do not invent policies, deadlines, prices or fact IDs.

Seed persona:
$user_persona

Seed goal:
$user_goal

Seed script:
$script

Return JSON with `user_persona`, `user_goal` and `script` (the beats as an
array of strings). English only. No real brand, person or company names.
