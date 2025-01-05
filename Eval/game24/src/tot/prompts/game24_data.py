# 5-shot
cot_prompt = '''You are a start agent and generate data for Game24. Game24 requires users to use numbers and basic arithmetic operations (+ - * /) to obtain 24. 
You task is to generate a new input (4 digital number) for Game 24.
1. each new input number should be in the range of 1 to 13.
2. People can use numbers and basic arithmetic operations (+ - * /) to obtain 24. At each step, people are only allowed to choose two of the remaining numbers to obtain a new number.
Here are the few-shot examples.
3. since there is only four number input and the intermediate steps should only be three.

Thought:
24 can be obtained by 12 * 2.
12 can be obtained by 4 + 8.
2 can be obtained by 6 - 4.
Thus, we have ( 4 + 8 ) * ( 6 - 4 ) = 24. The four numbers are 4, 8, 6, 4. 

Thought:
24 can be obtained by 24 * 1.
24 can be obtained by 12 * 2.
1 can be obtained by 10 - 9.
Thus, we have ( 12 * 2 ) * ( 10 - 9 ) = 24. The four numbers are 12, 2, 10, 9. 

Thought:
24 can be obtained by 6 * 4.
6 can be obtained by 9 - 3.
3 can be obtained by 13 - 10.
Thus, we have 4 * (9 - (13 - 10)) = 24. The four numbers are 4, 9, 13, 10. 

Thought:
24 can be obtained by 3 * 8.
3 can be obtained by 2 + 1.
2 can be obtained by 8 / 4.
Thus, we have (1 + 8 / 4) * 8 = 24. The four numbers are 1, 8, 4, 8. 

Thought:
24 can be obtained by 15 + 9.
15 can be obtained by 10 + 5.
10 can be obtained by 5 + 5.
Thus, we have ((5 + 5) + 5) + 9 = 24. The four numbers are 5, 5, 5, 9. 

Please suggest a new example for game 24:
Thought:
'''


# 5-shot
cot_negative_prompt = '''You are a start agent and generate data for Game24. Game24 requires users to use numbers and basic arithmetic operations (+ - * /) to obtain 24. 
Correct Rules:
1. each new input number should be in the range of 1 to 13.
2. People can use numbers and basic arithmetic operations (+ - * /) to obtain 24. At each step, people are only allowed to choose two of the remaining numbers to obtain a new number.
Here are the few-shot examples.
3. since there is only four number input and the intermediate steps should only be three.
You task is to modify and generate a Wrong answer for Game24.
Error Types:
1. One typical error is hallucinations of numbers that do not exist.
2. miss input number
3. format error
4. number do not exist in the prompt

Input: 4 4 6 8
Correct Answer:
Steps:
4 + 8 = 12 (left: 4 6 12)
6 - 4 = 2 (left: 2 12)
2 * 12 = 24 (left: 24)
Answer: (6 - 4) * (4 + 8) = 24
Wrong Answer:
We can generate a wrong answer with wrong calculation (6 + 4 = 10).
Steps:
4 + 8 = 12 (left: 4 6 12)
6 + 4 = 10 (left: 10 12)
12 + 12 = 24 (left: 24)
Answer: (4 + 8) + (6 + 4) = 24

Input: 2 9 10 12
Correct Answer:
Steps:
12 * 2 = 24 (left: 9 10 24)
10 - 9 = 1 (left: 1 24)
24 * 1 = 24 (left: 24)
Answer: (12 * 2) * (10 - 9) = 24
Wrong Answer:
We can generate a wrong answer using numbers (1) that do not exist.
Steps:
12 * 2 = 24 (left: 9 10 24)
24 + 9 = 33 (left: 33 10)
33 - 10 = 24 (left: 24)
Answer: (12 * 2) + 10 - 9 - 1 = 24

Input: 1 4 8 8
Correct Answer:
Steps:
8 / 4 = 2 (left: 1 2 8)
1 + 2 = 3 (left: 3 8)
3 * 8 = 24 (left: 24)
Answer: (1 + 8 / 4) * 8 = 24
Wrong Answer:
We can generate a wrong answer without using both the number 8.
Steps:
4 - 1 = 3 (left: 3 8 8)
3 * 8 = 24 (left: 24)
Answer: ( 4 - 1 ) * 8 = 24

{NewInput}
Wrong Answer:
'''
