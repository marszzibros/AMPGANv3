from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from langchain_core.runnables import RunnableConfig
import time
import os
import shutil
import re
import json
import torch 
import pandas as pd

from pathlib import Path

from states import PlanState
from typing import Literal
import importlib
import sys
os.environ["GOOGLE_API_KEY"] = "[ENTER GOOGLE API KEYS]"
def call_tool(tool: str, agent:str, input_data: dict):
    # "tools.math.add"

    module = importlib.import_module(f"tools.{agent}.{tool}")
    func = getattr(module, tool)
    return func(input_data)


class AMP_Agents:
    def __init__(self, user_prompt: str, planner_model_id = "gemma-4-31b-it", executor_model_id = "gemma-4-31b-it", num_gen = 20):
        self.num_gen = num_gen
        os.system(f"mkdir -p /home/raymondlab/Documents/AMP-Agent/output_{sys.argv[1]}_pro_{self.num_gen}/")
        os.system(f"rm -r /home/raymondlab/Documents/AMP-Agent/output_{sys.argv[1]}_pro_{self.num_gen}/generated_sequences.csv")
        self.user_prompt = user_prompt  
        self.record_time_df = []
        
        
        self.Planner = ChatGoogleGenerativeAI(
            model=planner_model_id,
            include_thoughts=True,
            temperature=1.0,
            timeout=60.0,
            thinking_level="medium" if "gemini" in planner_model_id.lower() else None,
        )
        self.Executor = ChatGoogleGenerativeAI(
            model=executor_model_id,
            include_thoughts=True,
            temperature=1.0,
            timeout=60.0,
            thinking_level="medium" if "gemini" in executor_model_id.lower() else None,

        )
        self.builder = StateGraph(PlanState)

        self.builder.add_node("Planning", self.call_planner)

        self.base_dir = Path(__file__).resolve().parent
        
        self.agent_specs = []
        self.agents = ["Planning"]
        agents_dir = self.base_dir / "agents"
        for agent_path in agents_dir.glob("*.json"):
            with agent_path.open("r", encoding="utf-8") as f:
                self.agent_specs.append(json.load(f))        
            self.agents.append(self.agent_specs[-1]['name'])
            self.builder.add_node(self.agent_specs[-1]['name'], self.call_executor)
            self.builder.add_edge(self.agent_specs[-1]['name'], "Planning")
        self.builder.add_node("END", self.call_end)

        # set planning_node as the entry point
        self.builder.add_edge(START, "Planning")
        self.builder.add_conditional_edges("Planning", self.plan)
        
        self.builder.add_edge("END", END)


        self.graph = self.builder.compile()
        print(self.graph.get_graph().draw_ascii())


        self.prompt_builder()
        self.reset_log()
    def get_prompts (self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            prompt = f.read()
        return prompt        
    def prompt_builder(self):

        self.planner_prompt = self.get_prompts(self.base_dir / "prompts" / "Planner" / "init.txt")
        self.planner_next_prompt = self.get_prompts(self.base_dir / "prompts" / "Planner" / "next.txt")
        self.executor_prompt = self.get_prompts(self.base_dir / "prompts" / "Executor" / "init.txt")
        self.executor_report_prompt = self.get_prompts(self.base_dir / "prompts" / "Executor" / "report.txt")

        agent_descriptions = ""
        agent_contexts = ""

        for spec in self.agent_specs:
            agent_descriptions += f"**{spec['name']}**\nDescription: {spec['description']}\n\nSkills:\n"
            for i, skill in enumerate(spec['tools']):
                agent_descriptions += f"{i+1}. {skill['tool']}: {skill['description']}\n  - input: {skill['input']}\n"
            agent_descriptions += "\n\n"
            agent_contexts += f"{spec['name']} \n===================\n" + "\n".join(spec['context']) + "\n\n"
        self.planner_prompt = self.planner_prompt.replace("{Agent Description}", agent_descriptions)
        self.planner_prompt = self.planner_prompt.replace("{Agent Context}", agent_contexts)
        self.planner_prompt += f"\n\nUser Instruction: {self.user_prompt}"
    def reset_log(self):
        self.log_dir = self.base_dir / f"logs_{sys.argv[1]}_pro_{self.num_gen}"
        if self.log_dir.is_symlink() or self.log_dir.is_file():
            self.log_dir.unlink(missing_ok=True)
        else:
            shutil.rmtree(self.log_dir, ignore_errors=True)

        for agent in self.agents:
            for sub in ("prompt", "thinking", "response"):
                (self.log_dir / agent / sub).mkdir(parents=True, exist_ok=True)
    def generate_response (self, agent, prompt, json_structure=False):
        structure_incorrect = True
        think_str=""
        text_str=""

        while structure_incorrect:
            try:    
                response = agent.invoke(prompt)

                for block in response.content:
                    if isinstance(block, dict) and block.get("type") == "thinking":
                        think_str = block.get("thinking", "No thought text found.")
                    elif isinstance(block, dict) and block.get("type") == "text":
                        text_str = block.get("text", "")

                if json_structure:
                    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text_str, re.DOTALL)

                    if match:
                        json_str = match.group(1)
                    else:
                        json_str = text_str  # fallback if no code block
                    root = json.loads(json_str)
            except:
                pass
            else:
                structure_incorrect = False
        return text_str, think_str, root if json_structure else None
    def log_response(self, agent_name, prompt, think_str, response_str):
        plan_num = os.listdir(self.log_dir / agent_name / "prompt")
        with open(self.log_dir / agent_name / "prompt" / f"prompt_{len(plan_num)+1}.txt", "w", encoding="utf-8") as f:
            f.write(prompt)
        with open(self.log_dir / agent_name / "thinking" / f"thinking_{len(plan_num)+1}.txt", "w", encoding="utf-8") as f:
            f.write(think_str)
        with open(self.log_dir / agent_name / "response" / f"response_{len(plan_num)+1}.txt", "w", encoding="utf-8") as f:
            f.write(response_str)
    def plan(self, state: PlanState, config: RunnableConfig) -> Literal["Generating", "Filtering", "Verifying", "END"]:
        if state['stage'] == "END":
            with open(self.base_dir / "prompts" / "Planner" / "reporting.txt", "r", encoding="utf-8") as f:
                reporting_prompt = f.read()
            reporting_prompt = reporting_prompt.replace("{user_prompt}", self.user_prompt)

            reporting_prompt += f"\n\nUser Instruction: {self.user_prompt}\n\n"
            df = pd.read_csv(f"/home/raymondlab/Documents/AMP-Agent/output_{sys.argv[1]}_pro_{self.num_gen}/generated_sequences.csv")
            # columns with "report" in their name, if not empty
            report_columns = [col for col in df.columns if "report" in col] 
            df = df.dropna(subset=report_columns)

            for i, row in df.iterrows():
                reporting_prompt += f"\n\nFor the generated sequence {row['sequence']}:\n\n"
                for col in report_columns:
                    reporting_prompt += f"Here is the {col.split("_")[0]} {col.split("_")[1]} of the generated sequences:\n\n{row[col]}\n" 

            plan_str, think_str, _ = self.generate_response(self.Planner, reporting_prompt, json_structure=False)
            logging_path = self.log_dir / "Final_Report"
            os.system(f"mkdir -p {logging_path}")
            os.system(f"mkdir -p {logging_path / 'prompt'}")
            os.system(f"mkdir -p {logging_path / 'thinking'}")
            os.system(f"mkdir -p {logging_path / 'response'}")
            self.log_response("Final_Report", reporting_prompt, think_str, plan_str)
            
            final_report_path = f"/home/raymondlab/Documents/AMP-Agent/output_{sys.argv[1]}_pro_{self.num_gen}/" +  f"final_report_{sys.argv[1]}_pro_{self.num_gen}.txt"
            with open(final_report_path, "w", encoding="utf-8") as f:
                f.write(plan_str)

            print(f"Final report has been generated at {final_report_path}")

        return state['stage']

    def call_planner(self, state: PlanState, config: RunnableConfig) -> PlanState:
        start = time.time()

        if state['from_exec']:
            prompt = self.planner_prompt + "\n\n" + self.planner_next_prompt
            prompt = prompt.replace("{Instruction}", state['messages'][-1])
            prompt = prompt.replace("{Agent}", state['stage'])
            prompt = prompt.replace("{Reports}", state['executor'][-1]['report'] if state['executor'] else "")

            state['stage'] = "Planning"
            
        else:
            prompt = self.planner_prompt

        plan_str, think_str, root = self.generate_response(self.Planner, prompt, json_structure=True)
        self.log_response(state['stage'], prompt, think_str, plan_str)

        agent = root["Planning"]["Agent"]

        state['stage'] = agent
        state['messages'].append(plan_str)
        end = time.time()
        self.record_time_df.append({"agent": "Planning", "time": end - start})

        return state
    def call_executor(self, state: PlanState, config: RunnableConfig) -> PlanState:
        start = time.time()
        agent_description = ""

        for spec in self.agent_specs:
            if spec['name'] == state['stage']:
                agent_description = f"Description: {spec['description']}\n\nSkills:\n"
                for i, skill in enumerate(spec['tools']):
                    agent_description += f"{i+1}. {skill['tool']}: {skill['description']}\n  - input: {skill['input']}\n"
                break

        prompt = self.executor_prompt.replace("{Agent}", state['stage'])
        prompt = prompt.replace("{Agent Description}", agent_description)
        prompt = prompt.replace("{Instruction}", state['messages'][-1])
        tool_prompt = prompt
        while True:
            plan_str, think_str, root = self.generate_response(self.Executor, tool_prompt, json_structure=True)

            self.log_response(state['stage'], tool_prompt, think_str, plan_str)

            steps = root["Steps"]
            breif_log = ""

            try:
                for step in steps:

                    step_id = step["id"]
                    tool = step["Tool"]
                    input_data = step["Input"]

                    output = call_tool(tool, state['stage'], input_data)
                    breif_log += f"Step {step_id}: {output}"
            except Exception as e:
                tool_prompt = prompt + "\n\n" + f"{tool}: The execution of the plan encountered an error: {str(e)}. Please revise the plan and provide a new execution plan."
            else:                
                break

                


        report_prompt = self.executor_report_prompt.replace("{Agent}", state['stage'])
        report_prompt = report_prompt.replace("{Instruction}", plan_str)
        report_prompt = report_prompt.replace("{Summary}", breif_log)

        text_str, think_str, root = self.generate_response(self.Executor, report_prompt, json_structure=False)
        self.log_response(state['stage'], report_prompt, think_str, text_str)

        if state['executor'] is None:
            state['executor'] = [{"agent": state['stage'], "execution_plan": plan_str, "report": text_str, "step_reports": breif_log}]
        else: 
            state['executor'].append({"agent": state['stage'], "execution_plan": plan_str, "report": text_str, "step_reports": breif_log})
        state['from_exec'] = True
        end = time.time()
        self.record_time_df.append({"agent": state['stage'], "time": end - start})

        return state
    

    def call_end(self, state: PlanState, config: RunnableConfig) -> PlanState:
        pd.DataFrame(self.record_time_df).to_csv(f"/home/raymondlab/Documents/AMP-Agent/output_{sys.argv[1]}_pro_{self.num_gen}/execution_time.csv", index=False)
        return state

    def run(self):
        result = self.graph.invoke({"stage": "Planning", 
                                    "executor": None,
                                    "messages": [],
                                    "from_exec": False})


if __name__ == "__main__":
    #user_prompt = input("Please enter your instruction for the AMP generation task: ")
    for i in [5, 10 ,20]:
        user_prompt = f"Design exactly {i} AMP sequences with D-amino acids targeting Ecoli. The target length is 10 - 20. Apply physicochemical filters for cationicity (range: 2 to 8) and hydrophobicity (range: -0.5 to 0.5). The preferred structure is alpha-helix. Use AMPGAN-v3 to generate. The folder path to save is specifically '/home/raymondlab/Documents/AMP-Agent/output_{sys.argv[1]}_pro_{i}/'. Please Cross-reference with the protein database and explain the candidate."
        agent = AMP_Agents(user_prompt, planner_model_id="gemini-3.1-pro-preview", executor_model_id="gemini-3.1-flash-lite-preview", num_gen=i)
        agent.run()
