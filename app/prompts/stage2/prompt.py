stage_2_prompt = """
Follow the step-by-step reasoning plan in JSON to correctly solve the task. Fill in the values following the keys by reasoning specifically about the task given. 
Do not simply rephrase the keys.
    
Reasoning Structure:
{reasoning_structure}

Task: {task_description}
"""