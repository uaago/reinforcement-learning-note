import numpy as np

np.random.seed=0
rewards=[]
'''
for n in range(1,11):
    reward=np.random.rand()
    rewards.append(reward)
    Q=sum(rewards)/n
    print(Q)
'''
#用Q直接更新
Q=0
for n in range(1,11):
    reward=np.random.rand()
    Q=Q+1/n*(reward-Q)
    print(Q)

