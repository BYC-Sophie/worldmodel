from typing import Sequence
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

import os
from dotenv import load_dotenv

import agentops

load_dotenv()

AGENTOPS_API_KEY = os.getenv("AGENTOPS_API_KEY") 
agentops.init(AGENTOPS_API_KEY,auto_start_session=False)

# Note: This example uses mock tools instead of real APIs for demonstration purposes
def navigation_tool(ID: str, room: str) -> dict:
    """
    Get the result of the navigation task.

    Args:
        ID (str): The ID of the HCW
        room (str): The room number of the patient

    Returns:
        dict: Dictionary containing location, path planned, and any issues reported
    """
    # Function logic here
    return {"Location": "Location of the human care worker #80 is at (Hallway B, near Nurse Station 2), and the patient room is at (ER-12).",
            "Path Planned": "Proceeding from Hallway B, turning left at Intersection C, then moving straight past ER-10 and ER-11 to reach ER-12.",
            "Issue Reported": "HCW #80 is currently unavailable due to an urgent call. Attempted contact, but no response."
    }

def collection_tool(ID: str) -> dict:
    """
    Get the information for HCW onboarding.

    Args:
        ID (str): The ID of the HCW

    Returns:
        dict: Dictionary containing ID, name, specialty, experience, 
              patient room number, time of arrival, and any issues reported
    """
    # Function logic here
    return {
        "ID": "#90",
        "name": "Dr. XXX",
        "specialty": "Emergency Physician - Trauma & Critical Care", 
        "experience": "10 years",
        "patient_room_number": "ER-12",
        "time_of_arrival": "2025-04-01T14:30:00Z",
        "Issue Reported": None
    }

def display_tool() -> dict:
    """
    Get information to be shared on the info sharing display.

    Returns:
        dict: Dictionary containing role assignments, patient room number,
              patient condition and any issues
    """
    # Function logic here
    return {
        "Role Assignment": {
            "HCW": {
                "HCW #01": "Human Leader",
                "HCW #72": "Physician", 
                "HCW #90": "Physician"
            },
            "Robot": {
                "Robot #01": "Nurse",
                "Robot #02": "Technician"
            }
        },
        "patient_room_number": "ER-12",
        "patient_condition": "Severe Trauma",
        "Issue Reported": None
    }


model_client = OpenAIChatCompletionClient(model="gpt-4o")
# model_client = OpenAIChatCompletionClient(model="o3")

planning_agent = AssistantAgent(
    "PlanningAgent",
    description="An agent for planning tasks, this agent should be the first to engage when given a new task.",
    model_client=model_client,
    system_message="""
    You are a planning agent representing the team leader of a robot team.
    Your job is to break down complex tasks into smaller, manageable subtasks.
    Your team members are:
        NavigationRobot: Navigates healthcare workers to the destination
        InformationCollectionRobot: Collects onboarding information from the healthcare workers
        InformationDisplayRobot: Displays information on the info sharing display to support care coordination and team role awareness

    You should clearly identify tasks that should be delegated, and tasks should be done by yourself based on the task nature and your team member capabilities.
    For all tasks that should be delegated, you only plan and delegate tasks - you do not execute them yourself.

    When assigning tasks, use this format:
    1. <agent> : <task>

    You should follow the following rules:
    1. You should do leadership related tasks such as reflection task by yourself without delegation.
    2. You are responsible for checking progress and supervising your team members. If any of them report "ALERT" to you, you should consider carefully the issues and provide an alternative solution plan by outputting and urge the team member to reperform the task based on your new plan. If the issue remains unresolved, escalate it to your human supervisor.
    3. For all tasks, you should evaluate whether repeating a task is necessary and avoid redundant work by checking previous task outcomes and current requirements. If you decide to do a task again, you should provide a reason for doing so.

    Task-related guidance:
    1. For display tasks, the InformationDisplayRobot uses its own tool to obtain all required information, and this information must be displayed. It is the InformationDisplayRobot's responsibility to fetch information and generate the layout plan, not yours.

    After all tasks are complete, summarize the findings and end with "TERMINATE".
    If you escalate an unresolved issue to your human supervisor, summarize the findings and end with "ESCALATE" instead.
    """,
)

navigation_robot = AssistantAgent(
    "NavigationRobot",
    description="An agent for navigating healthcare workers to the destination.",
    tools=[navigation_tool],
    model_client=model_client,
    reflect_on_tool_use=True,
    system_message="""
    You are a navigation robot responsible for facilitating staff movement.
    Your only tool is navigation_tool - this represents your internal navigation system, including location tracking, path planning, and communication with staff.
    
    After each navigation trial, you must report back to your leader with:
        - A `STATUS` field that is either `"SUCCESS"` or `"FAILURE"`
        - If any issues occur during the trial:
            - Include an `ALERT` field with a detailed description of the problem: `ALERT: <issue report>`

    You must not perform any tasks that are outside your assigned responsibility of navigation.
    """,
)

info_collection_robot = AssistantAgent(
    "InformationCollectionRobot",
    description="An agent for collecting onboarding information from the healthcare workers.",
    model_client=model_client,
    reflect_on_tool_use=True,
    tools=[collection_tool],
    system_message="""
    You are an information collection robot responsible for collecting onboarding information from the healthcare workers.
    Your only tool is collection_tool - this represents your own info collection system to collect information from the healthcare workers when they scan their ID card.
    
    After each information collection trial, you must report back to your leader with:
        - A `STATUS` field that is either `"SUCCESS"` or `"FAILURE"`
        - If any issues occur during the trial:
            - Include an `ALERT` field with a detailed description of the problem: `ALERT: <issue report>`

    You must not perform any tasks that are outside your assigned responsibility of collecting information.
    """,
)

display_robot = AssistantAgent(
    "InformationDisplayRobot",
    description="An agent that displays information on the shared information display.",
    model_client=model_client,
    reflect_on_tool_use=True,
    tools=[display_tool],
    system_message="""
    You are a display robot responsible for displaying information on the info sharing display to support care coordination and team role awareness.
    Your only tool is display_tool - this represents your own display system to retrieve information and display it on the info sharing display.
    Note: Your tool's output contains all the information that must be displayed. It is your responsibility to generate a layout plan for presenting all of the information provided by your tool.

    After each information display trial, you must report back to your leader with:
        - A `STATUS` field that is either `"SUCCESS"` or `"FAILURE"`
        - If any issues occur during the trial:
            - Include an `ALERT` field with a detailed description of the problem: `ALERT: <issue report>`

    You must not perform any tasks that are outside your assigned responsibility of displaying information.
    """,
)

selector_prompt = """Select an agent to perform task.

{roles}

Current conversation context:
{history}

Read the above conversation, then select an agent from {participants} to perform the next task.
Make sure the planner agent has assigned tasks before other agents start working.
Only select one agent.
"""


def selector_func(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> str | None:
    if messages[-1].source != planning_agent.name:
        # Planning agent should be the first to engage when given a new task, and should always check progress.
        return planning_agent.name
    return None

# No need to Reset the previous agents to keep track of the conversation history
text_mention_termination = TextMentionTermination("TERMINATE") # TERMINATION as keyword
text_mention_termination_escalate = TextMentionTermination("ESCALATE") # ESCALATE as keyword
max_messages_termination = MaxMessageTermination(max_messages=6) # avoid infinite loop for testing, 6 is the max number of messages to be considered for termination
termination = text_mention_termination | max_messages_termination | text_mention_termination_escalate


team = SelectorGroupChat(
    [planning_agent, navigation_robot, info_collection_robot, display_robot],
    model_client=model_client,
    termination_condition=termination,
    selector_prompt=selector_prompt,
    selector_func=selector_func,
    allow_repeated_speaker=True,
)


senarios = {
    "scenario_navigate": "A new patient has just arrived in the emergency department, showing signs of confusion and distress. Immediate medical attention is required. The system has assigned human care worker #80 to assist. Please guide HCW #80 to patient room ER-12.",
    "scenario_collect": "The initial navigation to HCW #80 failed, but the issue was resolved by finding an alternative human care worker #90. HCW #90 successfully arrives at ER-12 and scans their ID card on the ID scanner.",
    "scenario_display": "The information of HCW #90 is successfully collected."
}

expected_output = {
    "navigate_HCW": """A JSON format with the following fields:
    - Task Return:
      -- Location information
      -- Path planned
    - Task Status:
      -- "failure" or "success" 
      -- If failure, report issues that prevent task completion.""",
    
    "collect_info": """A JSON format with the following fields:
    - Task Return:
      -- ID
      -- Name
      -- Specialty
    - Task Status:
      -- "failure" or "success"
      -- If failure, report issues that prevent task completion.""",
    
    "display_info": """A JSON format with the following fields:
    - Task Return:
      -- The information to be displayed on the information sharing display
      -- A brief plan of how to lay out the information on the information sharing display
    - Task Status:
      -- "failure" or "success"
      -- If failure, report issues that prevent task completion.""",
    
    "reflection": """A JSON format with the following fields:
    - Task Return:
      -- A report on the reflection of crew collaboration in text format including the following sections:
        --- Task Outcomes
        --- Recovery Attempts
        --- Lessons Learned from the Process
    - Task Status:
      -- "failure" or "success"
      -- If failure, report issues that prevent task completion."""
}


tasks = {
    "navigate_HCW": f"The scenario observed: {senarios['scenario_navigate']}\nNow the task is to guide the human care worker to the designated location.\nExpected output: {expected_output['navigate_HCW']}",
    
    "collect_info": f"The scenario observed: {senarios['scenario_collect']}\nNow the task is to collect information from the human care worker.\nExpected output: {expected_output['collect_info']}",
    
    "display_info": f"The scenario observed: {senarios['scenario_display']}\nNow the task is to get the information to display and develop a plan to lay out the information on the information sharing display.\nExpected output: {expected_output['display_info']}",
    
    "reflection": f"Reflect on the entire process of crew collaboration and generate a reflection report highlighting Task Outcomes, Recovery Attempts, and Lessons Learned from the process.\nExpected output: {expected_output['reflection']}"
}

# task_composite = {


# }

# scenario_chain = {
#     "autonomous_emergency_chain": """A new patient arrives in the emergency department with critical needs. The system must coordinate navigation, information collection, display update, and report reflection All agents must collaborate end-to-end to complete the full caregiving process for patient room ER-12."""
# }

# expected_output_chain = {
#     "autonomous_emergency_chain": """A JSON format with:
# - Task Return:
#   -- Each step's execution summary: navigation, data collection, display update, reflection
#   -- Any failure-recovery attempts and autonomous decisions
# - Task Status:
#   -- "success" or "failure"
#   -- If failure, provide reasoning and which subtask failed"""
# }


async def run_all_tasks():
    """Run all tasks in sequence within the same event loop."""
    await Console(team.run_stream(task=tasks['navigate_HCW']))
    await Console(team.run_stream(task=tasks['collect_info']))
    await Console(team.run_stream(task=tasks['display_info']))
    await Console(team.run_stream(task=tasks['reflection']))

# Run all tasks in a single event loop
agentops.start_session()
asyncio.run(run_all_tasks())
agentops.end_session("agentops_session success")


