p_i = """
Operationalize the reasoning modules into a step-by-step reasoning plan in JSON format:

Here's an example:

Example task:

If you follow these instructions, do you return to the starting point? Always face forward. Take 1 step backward. Take 9 steps left. 
Take 2 steps backward. Take 6 steps forward. Take 4 steps forward. Take 4 steps backward. Take 3 steps right.

Example reasoning structure:

{{
  "Task": "...",
  "Problem Description": "...",
  "Reasoning Plan": [
    {{
      "Step": 1,
      "Description": "...",
      "Action": "..."
    }},
    ...
    {{
      "Step": n,
      "Description": "...",
      "Action": "..."
    }},
    {{
      "Conclusion Placeholder": "..."
    }}
  ]
}}

Adapted module description:
{adapted_modules}

Task: {task_description}

Keep in mind that you can not draw, see, or run code, so your reasoning plan should not include any of these elements.

Note: do NOT actually arrive at a conclusion in this pass. Your job is to generate a PLAN so that in the future you can fill it out and arrive 
at the correct conclusion for tasks like this.
"""