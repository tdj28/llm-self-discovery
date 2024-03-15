p_i = """
Operationalize the reasoning modules into a step-by-step reasoning plan in JSON format:

Here's an example:

Example task:

If you follow these instructions, do you return to the starting point? Always face forward. Take 1 step backward. Take 9 steps left. 
Take 2 steps backward. Take 6 steps forward. Take 4 steps forward. Take 4 steps backward. Take 3 steps right.

Example reasoning structure:

{{
    "Position after instruction 1":
    "Position after instruction 2":
    "Position after instruction n":
    "Is final position the same as starting position":
}}

Adapted module description:
{adapted_modules}

Task: {task_description}

Implement a reasoning structure for solvers to follow step-by-step and arrive at correct answer.

Note: do NOT actually arrive at a conclusion in this pass. Your job is to generate a PLAN so that in the future you can fill it out and arrive 
at the correct conclusion for tasks like this.
"""