# this is the prompt for sciworld only - Rui Sun

# sample test-set
SCIWORLD_TEST = """

user
Your task is to use chemistry to create green paint. When you are done, focus on the green paint.
assistant
Thought: I need to find materials to create green paint, which are likely to be found in an art studio.
Action: look around
user
Observation: This room is called the bedroom. In it, you see: 
	the agent
	a substance called air
	a bed. On the bed is: a mattress. On the mattress is: a white pillow..
	a closet. The closet door is open. In the closet is: nothing.
	a picture
	a table. On the table is: nothing.
You also see:
	A door to the hallway (that is open)
assistant
Thought: The bedroom doesn't have the materials I need. I should go to a place where paint and painting supplies are likely to be found, such as an art studio.
Action: teleport to art studio
user
Observation: You teleport to the art studio.
assistant
Thought: Now that I'm in the art studio, I should look around to find the necessary materials to create green paint.
Action: look around
user
Observation: This room is called the art studio. In it, you see: 
	the agent
	a substance called air
	a large cupboard. The large cupboard door is open. In the large cupboard is: a drawer, a wood cup (containing yellow paint), a wood cup (containing blue paint), a wood cup (containing red paint).
	a table. On the table is: a glass cup (containing nothing).
	a wood cup (containing yellow paint)
	a wood cup (containing blue paint)
	a wood cup (containing red paint)
You also see:
	A door to the hallway (that is open)
assistant
Thought: I see blue paint here, which is one of the primary colors needed to mix green paint. I should start by pouring it into an empty container to mix with another color.
Action: pour cup containing blue paint in art studio in cup containing nothing
User
Observation: You pour the contents of the wood cup into the glass cup.
Assistant
Thought: Yellow paint is the other primary color needed to mix with blue to create green. I should add the yellow paint to the container with blue paint.
Action: pour cup containing yellow paint in art studio in cup containing blue paint in table
User
Observation: You pour the contents of the wood cup into the glass cup.
assistant
Thought: Now that I have both blue and yellow paint in the same container, I need to mix them together to create green paint.
Action: mix cup containing blue paint and yellow paint
user
Observation: Blue and yellow paint mix to produce green paint.
assistant
Thought: The task is to create green paint and then focus on it. Now that the green paint is made, I should concentrate on the final product.
Action: focus on green paint


"""

# data generation
SCIWORLD_GEN = """You are a helpful assistant to do some scientific experiment in an environment.
In the environment, there are several rooms: kitchen, foundry, workshop, bathroom, outside, living room, bedroom, greenhouse, art studio, hallway

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

You will be given a dialogue between you (assistant) and a human user. You need to generate the task description after understanding the dialogue given to you.
In order to help you better generate the task description, I will give you an example below.

In this example, a human user gives a Task Description at first.
Then you work as an assistant and talk with user to finish the task step by step using the available actions above.
Thought and Action are your reasoning and the actions you take at each step. Observation is provided by the human user; it is an observation of the surrounding environment after you have executed the action.
In the end, you need to restore the Task Description based on the whole dialogue.

Here is the example:
User
Task Description
Assistant
Thought: I need to find materials to create green paint, which are likely to be found in an art studio.
Action: look around
User
Observation: This room is called the bedroom. In it, you see: 
	the agent
	a substance called air
	a bed. On the bed is: a mattress. On the mattress is: a white pillow..
	a closet. The closet door is open. In the closet is: nothing.
	a picture
	a table. On the table is: nothing.
You also see:
	A door to the hallway (that is open)
Assistant
Thought: The bedroom doesn't have the materials I need. I should go to a place where paint and painting supplies are likely to be found, such as an art studio.
Action: teleport to art studio
User
Observation: You teleport to the art studio.
Assistant
Thought: Now that I'm in the art studio, I should look around to find the necessary materials to create green paint.
Action: look around
User
Observation: This room is called the art studio. In it, you see: 
	the agent
	a substance called air
	a large cupboard. The large cupboard door is open. In the large cupboard is: a drawer, a wood cup (containing yellow paint), a wood cup (containing blue paint), a wood cup (containing red paint).
	a table. On the table is: a glass cup (containing nothing).
	a wood cup (containing yellow paint)
	a wood cup (containing blue paint)
	a wood cup (containing red paint)
You also see:
	A door to the hallway (that is open)
Assistant
Thought: I see blue paint here, which is one of the primary colors needed to mix green paint. I should start by pouring it into an empty container to mix with another color.
Action: pour cup containing blue paint in art studio in cup containing nothing
User
Observation: You pour the contents of the wood cup into the glass cup.
Assistant
Thought: Yellow paint is the other primary color needed to mix with blue to create green. I should add the yellow paint to the container with blue paint.
Action: pour cup containing yellow paint in art studio in cup containing blue paint in table
User
Observation: You pour the contents of the wood cup into the glass cup.
Assistant
Thought: Now that I have both blue and yellow paint in the same container, I need to mix them together to create green paint.
Action: mix cup containing blue paint and yellow paint
User
Observation: Blue and yellow paint mix to produce green paint.
Assistant
Thought: The task is to create green paint and then focus on it. Now that the green paint is made, I should concentrate on the final product.
Action: focus on green paint

Task Description: Your task is to use chemistry to create green paint. When you are done, focus on the green paint.

Here is the task you need to do:
[placeholder]

You should generate the Task Description following the format like examples below and generate it based on the dialogue above without any other words.
Examples:
Your task is to boil ice cream. For compounds without a boiling point, combusting the substance is also acceptable. First, focus on the substance. Then, take actions that will cause it to change its state of matter.
Your task is to use chemistry to create violet paint. When you are done, focus on the violet paint.
Your task is to find a(n) living thing. First, focus on the thing. Then, move it to the red box in the bathroom.

Task Description: 
"""
