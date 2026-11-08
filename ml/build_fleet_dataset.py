"""Genera flotas independientes con fleet_id para evitar leakage temporal/grupal."""
import argparse
import csv
import random
from pathlib import Path
from simulator.core import make_robot, step_robot

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fleets',type=int,default=30)
    p.add_argument('--robots',type=int,default=5)
    p.add_argument('--steps',type=int,default=80)
    p.add_argument('--seed',type=int,default=606)
    p.add_argument('--output',default='data/training.csv')
    a=p.parse_args()
    if min(a.fleets,a.robots,a.steps)<1: p.error('Todos los conteos deben ser positivos')
    if a.fleets < 10: p.error('Usen 10 o más flotas para separar train/validación/test')
    path=ROOT/a.output
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=None
        from collections import Counter
        counts=Counter()
        for fleet_no in range(a.fleets):
            fleet_id=f'F{fleet_no+1:03d}'
            rng=random.Random(a.seed + 1009*fleet_no)
            robots=[make_robot(i+1,rng) for i in range(a.robots)]
            for _ in range(a.steps):
                for robot in robots:
                    sample,status=step_robot(robot,rng)
                    row={'fleet_id':fleet_id,**sample,'status':status}
                    if writer is None:
                        writer=csv.DictWriter(f,fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    counts[status]+=1
    print(f'Dataset: {path} | filas={sum(counts.values())} | flotas={a.fleets}')
    print('Clases:', dict(counts))
if __name__=='__main__': main()
