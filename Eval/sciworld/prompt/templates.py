import os
import json

PROMPT_FIX = """
Please only output one Thought and one Action.
The available actions are:
open OBJ: open a container
close OBJ: close a container
activate OBJ: activate a device
deactivate OBJ: deactivate a device
connect OBJ to OBJ: connect electrical components
disconnect OBJ: disconnect electrical components
use OBJ [on OBJ]: use a device/item
look around: describe the current room
examine OBJ: describe an object in detail
look at OBJ: describe a container's contents
read OBJ: read a note or book
move OBJ to OBJ: move an object to a container
pick up OBJ: move an object to the inventory
pour OBJ into OBJ: pour a liquid into a container
mix OBJ: chemically mix a container
teleport to LOC: teleport to a specific room
focus on OBJ: signal intent on a task object
wait: task no action for 10 steps
wait1: task no action for a step
Please strictly follow the format of the actions.
"""
PROMPT_WITH_ICL_TEMPLATE = """{instruction}
---
{icl_prompt}

{examples}
---

Now, it's your turn and here is the task.
{task}

"""

PROMPT_WITH_ICL_TEMPLATE += PROMPT_FIX



def prompt_with_icl(instruction, raw_icl, cur_task, icl_num=2):
    examples = ""
    messages = [{
        "role": "user",
        "content": instruction
    }]
    for i in range(min(icl_num, len(raw_icl))):
        for j in range(len(raw_icl[i])):
            cur_content = raw_icl[i][j]['content']
            if i == 0 and j == 0:
                messages.append({
                    "role": "assistant",
                    "content": "OK"
                })
                messages.append({
                    "role": "user",
                    "content": cur_content
                })
                if icl_num > 1:
                    examples += f"Example task {i + 1}:\n"
                examples += cur_content + '\n'
                continue
            elif i != 0 and j == 0:
                if icl_num > 1:
                    examples += f"\nExample task {i + 1}:\n"
                    examples += cur_content + '\n'
                else:
                    examples += '\n' + cur_content + '\n'
                messages.append({
                    "role": "user",
                    "content": cur_content
                })
                continue
            # user
            if j % 2 == 0:
                examples += cur_content + '\n\n'
                messages.append({
                    "role": "user",
                    "content": cur_content
                })
            # assistant
            else:
                examples += cur_content + '\n'
                messages.append({
                    "role": "assistant",
                    "content": cur_content
                })
    icl_prompt = f"Here are {icl_num} examples." if icl_num > 1 else f"Here is an example."
    prompt = PROMPT_WITH_ICL_TEMPLATE.format(instruction=instruction, icl_prompt=icl_prompt, examples=examples, task=cur_task)
    messages.append({
        "role": "user",
        "content": cur_task + PROMPT_FIX
    })

    return prompt, messages
