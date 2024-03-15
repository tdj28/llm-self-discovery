from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional  

from app.prompts.stage1 import p_s, p_a, p_i
from app.prompts.stage2 import stage_2_prompt
from app.reasoning_modules import reasoning_modules

model = ChatOpenAI(temperature=0.5, model="gpt-4-turbo-preview")

########### STAGE 1 ###########
select_prompt = PromptTemplate(input_variables=["reasoning_modules", "task_description"], template=p_s )
adapt_prompt = PromptTemplate(input_variables=["selected_modules", "task_description"], template=p_a )
implement_prompt = PromptTemplate(input_variables=["adapted_modules", "task_description"], template=p_i )

########### STAGE 2 ###########
stage_2_prompt_ = PromptTemplate(input_variables=["reasoning_structure", "task_description"], template=stage_2_prompt)

########### Graph #############

class SelfDiscoverState(TypedDict):
    reasoning_modules: str
    task_description: str
    selected_modules: Optional[str]
    adapted_modules: Optional[str]
    reasoning_structure: Optional[str]
    answer: Optional[str]

def select(inputs):
    select_chain = select_prompt | model | StrOutputParser()
    return {"selected_modules": select_chain.invoke(inputs)}

def adapt(inputs):
    adapt_chain = adapt_prompt | model | StrOutputParser()
    return {"adapted_modules": adapt_chain.invoke(inputs)}

def implement(inputs):
    implement_chain = implement_prompt | model | StrOutputParser()
    return {"reasoning_structure": implement_chain.invoke(inputs)}

def stage_2(inputs):
    stage_2_chain = stage_2_prompt_ | model | StrOutputParser()
    return {"answer": stage_2_chain.invoke(inputs)}

graph = StateGraph(SelfDiscoverState)
graph.add_node("select", select)
graph.add_node("adapt", adapt)
graph.add_node("implement", implement)
graph.add_node("stage_2", stage_2)
graph.add_edge("select", "adapt")
graph.add_edge("adapt", "implement")
graph.add_edge("implement", "stage_2")
graph.add_edge("stage_2", END)
graph.set_entry_point("select")
app = graph.compile()


task_example = """This SVG path element <path d="M 55.57,80.69 L 57.38,65.80 M 57.38,65.80 L 48.90,57.46 M 48.90,57.46 L
45.58,47.78 M 45.58,47.78 L 53.25,36.07 L 66.29,48.90 L 78.69,61.09 L 55.57,80.69"/> draws a:
(A) circle (B) heptagon (C) hexagon (D) kite (E) line (F) octagon (G) pentagon(H) rectangle (I) sector (J) triangle"""

reasoning_modules_str = "\n".join(reasoning_modules)

print("\n\n\n")
for s in app.stream(
    {"task_description": task_example, "reasoning_modules": reasoning_modules_str}
):
    print(f"Current state: {repr(s).replace('\\n', '\n')}")
    print("\n\n\n")