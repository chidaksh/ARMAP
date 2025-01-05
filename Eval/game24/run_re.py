import os
import json
import argparse

from tot.tasks import get_task
from tot.methods.bfs_re import solve, naive_solve, get_re
from tot.models import gpt_usage
import pdb
from tot.argparser import parse_args
from tot.methods.mcts_agents.utils_mcts import call_api

def set_debugger():
    from IPython.core import ultratb
    import sys
    sys.excepthook = ultratb.FormattedTB(call_pdb=True)
set_debugger()

def get_reward(model, ys, infos, x):
    if model == 'gt':
        return infos[0]['r']
    else:
        trajectory=ys[0]
        if 'Steps:' in trajectory:
            trajectory = 'Steps:' + trajectory.split('Steps:')[-1]
        else:
            trajectory = 'Steps:\n' + trajectory
        return call_api(model, state=dict(input=x, trajectory=trajectory), num_api=1)


def run(args):
    task = get_task(args.task)
    logs, cnt_avg, cnt_any = [], 0, 0
    if args.naive_run:
        file = f'./logs/{args.task}/re_iter{args.re_iterations}_{args.backend}_{args.temperature}_reward_{args.reward}_start{args.task_start_index}_end{args.task_end_index}.json'
    os.makedirs(os.path.dirname(file), exist_ok=True)

    for i in range(args.task_start_index, args.task_end_index):

        history = None
        for iter in range(args.re_iterations):
            
            # solve
            if args.naive_run:
                ys, info, x = naive_solve(args, task, i, history=history)
            # else:
            #     ys, info = solve(args, task, i, to_print=True)

            # log
            print(ys[0])
            infos = [task.test_output(i, y) for y in ys]
            reward = get_reward(args.reward, ys, infos, x)
            if reward > 0:
                break
            if not history:
                history = []
            history.append(get_re(args, task, i, ys))



        info.update({'idx': i, 'ys': ys, 'infos': infos, 'usage_so_far': gpt_usage(args.backend)})


        logs.append(info)
        with open(file, 'w') as f:
            json.dump(logs, f, indent=4)
        
        # log main metric
        accs = [info['r'] for info in infos]
        cnt_avg += sum(accs) / len(accs)
        cnt_any += any(accs)
        print(i, 'sum(accs)', sum(accs), 'cnt_avg', cnt_avg, 'cnt_any', cnt_any, '\n')
    
    n = args.task_end_index - args.task_start_index
    print(cnt_avg / n, cnt_any / n)
    print('usage_so_far', gpt_usage(args.backend))




if __name__ == '__main__':
    args = parse_args()
    print(args)
    run(args)
