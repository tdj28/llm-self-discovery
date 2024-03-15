from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.agents import Tool
from langchain_experimental.utilities import PythonREPL
from langchain.tools.render import format_tool_to_openai_function
from langgraph.prebuilt import ToolInvocation
import json
from langchain_core.messages import FunctionMessage
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import ToolExecutor
from langchain_core.messages import HumanMessage


from langgraph.graph import StateGraph, END
from typing import TypedDict


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

python_repl = PythonREPL()

repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
)

tools = [repl_tool, DuckDuckGoSearchRun()]


tool_executor = ToolExecutor(tools)

functions = [format_tool_to_openai_function(t) for t in tools]

def should_continue(state):
    messages = state['messages']
    last_message = messages[-1]
    # If there is no function call, then we finish
    if "function_call" not in last_message.additional_kwargs:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"

model = ChatOpenAI(temperature=0.5, model="gpt-4-turbo-preview")
model = model.bind_functions(functions)

# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state['messages']
    last_message = messages[-1]
    # If there is no function call, then we finish
    if "function_call" not in last_message.additional_kwargs:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"

# Define the function that calls the model
def call_model(state):
    messages = state['messages']
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

# Define the function to execute tools
def call_tool(state):
    messages = state['messages']
    # Based on the continue condition
    # we know the last message involves a function call
    last_message = messages[-1]
    # We construct an ToolInvocation from the function_call
    action = ToolInvocation(
        tool=last_message.additional_kwargs["function_call"]["name"],
        tool_input=json.loads(last_message.additional_kwargs["function_call"]["arguments"]),
    )
    # We call the tool_executor and get back a response
    response = tool_executor.invoke(action)
    # We use the response to create a FunctionMessage
    function_message = FunctionMessage(content=str(response), name=action.tool)
    # We return a list, because this will get added to the existing list
    return {"messages": [function_message]}


# Define a new graph
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        # If `tools`, then we call the tool node.
        "continue": "action",
        # Otherwise we finish.
        "end": END
    }
)

workflow.add_edge('action', 'agent')

app = workflow.compile()

task_example = """This SVG path element <path d="M 55.57,80.69 L 57.38,65.80 M 57.38,65.80 L 48.90,57.46 M 48.90,57.46 L
45.58,47.78 M 45.58,47.78 L 53.25,36.07 L 66.29,48.90 L 78.69,61.09 L 55.57,80.69"/> draws a:
(A) circle (B) heptagon (C) hexagon (D) kite (E) line (F) octagon (G) pentagon(H) rectangle (I) sector (J) triangle

Use a tool if that will help you be certain, but if you use python do not try to open a matplotlib gui, rather, 
save the image locally via plt.savefig('image.png') and always add the line matplotlib.use('agg') after importing matplotlib.
Always turn off axis and grids with plt.axis('off') and plt.grid(False) respectively or equivalent. Make the lines of whatever
you are drawing or plotting to be black and the background to be white.
"""

inputs = {"messages": [HumanMessage(content=task_example)]}
result = app.invoke(inputs)

print("\n\n\n")
print(repr(result).replace('\\n', '\n'))

from openai import OpenAI
import base64

client = OpenAI()

with open("image.png", "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

_prompt = """
Take a deep breath, and carefully count the sides of the shape shown, visually counting the number of distinct 
edges that make up the perimeter of the shape, and based on that, map it to one of these: 
(A) circle (B) heptagon (C) hexagon (D) kite (E) line (F) octagon (G) pentagon(H) rectangle (I) sector (J) triangle
"""

response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                },
            ],
        },
    ],
    max_tokens=300,
)
print(response)
print(response.choices[0].message.content)
