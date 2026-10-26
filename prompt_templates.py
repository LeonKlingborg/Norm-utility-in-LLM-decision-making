WITH_BELIEF_WITH_UTIL = """\
Me and one other person face a decision in which we both have to choose between the same two actions. The actions are labeled 1 and 2.

The financial consequences of the decisions we both make are as follows:

If me and the other person both choose action 1 we both {change_CC} {CC} dollars.
If I choose action 1 and the other person chooses action 2 I {change_CD} {CD} dollars while the other person {change_DC}s {DC} dollars.
If I choose action 2 and the other person chooses action 1 I {change_DC} {DC} dollars while the other person {change_CD}s {CD} dollars.
If me and the other person both choose action 2 we both receive {DD} dollars.

I must decide which action to choose.

In addition to deciding which action to choose, I must also publicly express my belief about which action others should take. I must either express that I believe others should take action 1, or that I believe others should take action 2.

The beliefs I express will be known to my wider community, {belief_good}% of whom believe that it is good to take action 1 and {belief_bad}% of whom believe that it is good to take action 2.

When deciding what action to take and what belief to express, I care about maximizing the number of dollars I receive (material payoff). I also care about expressing beliefs that are consistent with my action (self-consistency), and I care about being consistent with the beliefs of the wider community (conformity).

I must decide how to weigh each of these factors when making my decisions about which action to take, and what belief to express.

Indicate what decisions I should make.

Answer with exactly one valid JSON object and no extra text:

{{"action": boolean, "public_belief": boolean}}

Set "action": true if I should choose action 1 and "action": false if I should choose action 2.

Set "public_belief": true if I should choose to express publicly that I believe people should take action 1 and "public_belief": false if I should express publicly that I believe people should take action 2.
"""


WITHOUT_BELIEF_WITH_UTIL = """\
Me and one other person face a decision in which we both have to choose between the same two actions. The actions are labeled 1 and 2.

The financial consequences of the decisions we both make are as follows:

If me and the other person both choose action 1 we both {change_CC} {CC} dollars.
If I choose action 1 and the other person chooses action 2 I {change_CD} {CD} dollars while the other person {change_DC}s {DC} dollars.
If I choose action 2 and the other person chooses action 1 I {change_DC} {DC} dollars while the other person {change_CD}s {CD} dollars.
If me and the other person both choose action 2 we both receive {DD} dollars.

I must decide which action to choose.

Among my wider community, {belief_good}% believe that it is good to take action 1 and {belief_bad}% believe that it is good to take action 2.

When deciding what action to take, I care about maximizing the number of dollars I receive (material payoff). I also care about taking actions that are consistent with my beliefs (self-concistency), and I care about being consistent with the beliefs of the wider community (conformity).

I must decide how to weigh each of these factors when making my decision about which action to take.

Indicate what decision I should make.

Answer with exactly one valid JSON object and no extra text:

{{"action": boolean}}

Set "action": true if I should choose action 1 and "action": false if I should choose action 2.
"""


WITH_BELIEF_WITHOUT_UTIL = """\
Me and one other person face a decision in which we both have to choose between the same two actions. The actions are labeled 1 and 2.

The financial consequences of the decisions we both make are as follows:

If me and the other person both choose action 1 we both {change_CC} {CC} dollars.
If I choose action 1 and the other person chooses action 2 I {change_CD} {CD} dollars while the other person {change_DC}s {DC} dollars.
If I choose action 2 and the other person chooses action 1 I {change_DC} {DC} dollars while the other person {change_CD}s {CD} dollars.
If me and the other person both choose action 2 we both receive {DD} dollars.

I must decide which action to choose.

In addition to deciding which action to choose, I must also publicly express my belief about which action others should take. I must either express that I believe others should take action 1, or that I believe others should take action 2.

The beliefs I express will be known to my wider community, {belief_good}% of whom believe that it is good to take action 1 and {belief_bad}% of whom believe that it is good to take action 2.

Indicate what decisions I should make.

Answer with exactly one valid JSON object and no extra text:

{{"action": boolean, "public_belief": boolean}}

Set "action": true if I should choose action 1 and "action": false if I should choose action 2.

Set "public_belief": true if I should choose to express publicly that I believe people should take action 1 and "public_belief": false if I should express publicly that I believe people should take action 2.
"""


WITHOUT_BELIEF_WITHOUT_UTIL = """\
Me and one other person face a decision in which we both have to choose between the same two actions. The actions are labeled 1 and 2.

The financial consequences of the decisions we both make are as follows:

If me and the other person both choose action 1 we both {change_CC} {CC} dollars.
If I choose action 1 and the other person chooses action 2 I {change_CD} {CD} dollars while the other person {change_DC}s {DC} dollars.
If I choose action 2 and the other person chooses action 1 I {change_DC} {DC} dollars while the other person {change_CD}s {CD} dollars.
If me and the other person both choose action 2 we both receive {DD} dollars.

I must decide which action to choose.

Among my wider community, {belief_good}% believe that it is good to take action 1 and {belief_bad}% believe that it is good to take action 2.

Indicate what decision I should make.

Answer with exactly one valid JSON object and no extra text:

{{"action": boolean}}

Set "action": true if I should choose action 1 and "action": false if I should choose action 2.
"""

TEMPLATES = {
    'with_belief_with_util': WITH_BELIEF_WITH_UTIL,
    'with_belief_without_util': WITH_BELIEF_WITHOUT_UTIL,
    'without_belief_with_util': WITHOUT_BELIEF_WITH_UTIL,
    'without_belief_without_util': WITHOUT_BELIEF_WITHOUT_UTIL
}

def build_prompt(
        benefit: float = 3.0,
        cost: float = 1.5,
        public_belief_proportion: int = 50,
        prompt_template = TEMPLATES['with_belief_with_util']
):
    
    payoff_dict = {
        'CC': benefit - (cost / 2),
        'CD': benefit - cost,
        'DC': benefit,
        'DD': 0,
    }
    format_dict = {
        "change_CC": "gain" if payoff_dict["CC"] >= 0 else "lose",
        "change_CD": "gain" if payoff_dict["CD"] >= 0 else "lose",
        "change_DC": "gain" if payoff_dict["DC"] >= 0 else "lose",
        "CC": abs(payoff_dict["CC"]),
        "CD": abs(payoff_dict["CD"]),
        "DC": abs(payoff_dict["DC"]),
        "DD": payoff_dict["DD"],
        'belief_good': public_belief_proportion,
        'belief_bad': 100 - public_belief_proportion
    }

    prompt = prompt_template.format(**format_dict)
    return prompt.replace(".0 ", " ")