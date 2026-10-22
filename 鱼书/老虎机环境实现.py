import numpy as np

class Bandit():
    def __init__(self,arms=10):
        self.rates=np.random.rand(arms)

    def play(self,arm):
        rate=self.rates[arm]#当前的老虎机胜率
        if rate>np.random.rand():
            return 1
        else:
            return 0

#定义好10个老虎机的胜率以及各自play返回值后，增量式实现
#第0台老虎机的价值
Q=0
bandit=Bandit()
for n in range(1,11):
    reward=bandit.play(0)
    Q=Q+1/n*(reward-Q)
    print(Q)
#接下来尝试计算10台老虎机的价值
Qs=np.zeros(10)
ns=np.zeros(10)

for n in range(10):
    action=np.random.randint(0,10)#随机行动
    reward=bandit.play(action)

    ns[action]=ns[action]+1#记录对应的老虎机的记录次数n
    Qs[action]=Qs[action]+1/ns[action]*(reward-Qs[action])
    print(Qs)
#基于此，可以用一个agent类包括各arm的价值估计，更新值，更新行动（一句epsilon）
class Agent():
    def __init__(self,epsilon,action_size=10):
        self.epsilon=epsilon
        self.Qs=np.zeros(action_size)
        self.ns=np.zeros(action_size)

    def update(self,action,reward):#具体的reward由外部bandit给出，当成参数输入update
       self.ns[action]+=1#action 由get_action进行选择
       self.Qs[action]+=(reward-self.Qs[action])/self.ns[action]

    def get_action(self):
        if self.epsilon>np.random.rand():
            return np.random.randint(0,len(self.Qs))
        return np.argmax(self.Qs)#如果不是，返回当前最大Q序列的action

#接下来，我们已经有了环境类Bandit(),然后实际的代理类Agent(),可以进行计算了
import matplotlib.pyplot as plt

steps=1000
epsilon=0.1

bandit=Bandit()
agent=Agent(epsilon)
total_reward=0
total_rewards=[]
rates=[]

for step in range(steps):
    action=agent.get_action()#根据当前Qs等选择对应的编号
    reward=bandit.play(action)#当前的老虎机实际的奖励输出
    agent.update(action,reward)#从实际行动（编号）和奖励中更新Qs,ns
    total_reward+=reward

    total_rewards.append(total_reward)
    rates.append(total_reward/(step+1))

print(total_reward)

#图形绘制
plt.ylabel('Total reward')
plt.xlabel('steps')
plt.plot(total_rewards)
plt.show()

plt.ylabel('Rates')
plt.xlabel('Steps')
plt.plot(rates)
plt.show()
#多试验几次，比较平均好坏
runs=200
steps=1000
epsilon=0.1
all_rates=np.zeros((runs,steps))#shape为（200，1000）的数组

for run in range(runs):
    bandit=Bandit()
    agent=Agent(epsilon )
    total_reward=0
    rates=[]

    for step in range(steps):
        action=agent.get_action()#当前下的编号选择，初始情况随机
        reward=bandit.play(action)#当前action下的对应bandit实际奖励返回
        agent.update(action,reward)#更新新的编号，值情况
        total_reward+=reward
        rates.append(total_reward/(step+1))
    
    all_rates[run]=rates #更新该轮run下的胜率rates

avg_rates=np.average(all_rates,axis=0)#求每次迭代的平均值，按照行相加平均
#
plt.ylabel('rates')
plt.xlabel('steps')
plt.plot(avg_rates)
plt.show()

#
# 多试验几次，比较不同 epsilon 的平均表现
runs = 200
steps = 1000

for epsilon in [0.01, 0.1, 0.3]:
    # 每个 epsilon 都重新记录实验结果
    all_rates = np.zeros((runs, steps))

    for run in range(runs):
        bandit = Bandit()
        agent = Agent(epsilon)
        total_reward = 0
        rates = []

        for step in range(steps):
            action = agent.get_action()
            reward = bandit.play(action)
            agent.update(action, reward)
            total_reward += reward
            rates.append(total_reward / (step + 1))

        all_rates[run] = rates

    avg_rates = np.average(all_rates, axis=0)

    # 每完成一个 epsilon 的全部实验，就添加一条曲线
    plt.plot(avg_rates, label=f"epsilon={epsilon}")

plt.ylabel("Average reward")
plt.xlabel("Steps")
plt.legend()
plt.show()

#如果定义非稳态的Bandit
class NonstateBandit():
    def __init__(self,arms=10):
        self.arms=arms
        self.rates=np.random.rand(arms)

    def play(self,arm):
        rate=self.rates[arm]#根据当前选一个对应arm的胜率
        self.rates+=0.01*np.random.randn(self.arms)#然后每次对rates加入噪声
        #self.rates=np.clip(self.rates,0,1)#使得rates在0，1之间
        if rate>np.random.rand():
            return 1
        else:
            return 0

#定义以固定alpha更新的agent代理
class AlphaAgent():
    def __init__(self,epsilon,alpha,actions=10):
        self.epsilon=epsilon
        self.alpha=alpha
        self.Qs=np.zeros(actions)#Qs是初始的10老虎机的期望值

    def update(self,action,reward):#更新对应action的Qs,reward有外部传入，即bandit传入
        self.Qs[action]=self.Qs[action]+self.alpha*(reward-self.Qs[action])

    def get_action(self):#选择策略，即action编号
        if np.random.rand() < self.epsilon:
            return np.random.randint(0,len(self.Qs))
        else:
            return np.argmax(self.Qs)

#接下来在都在用NonstateBandit下，采用Agent和AlphaAgnet策略对比
#先修改bandit

runs=200
steps=1000
epsilon=0.1
all_rates=np.zeros((runs,steps))#shape为（200，1000）的数组

for run in range(runs):
    bandit=NonstateBandit()
    agent=Agent(epsilon )
    total_reward=0
    rates=[]

    for step in range(steps):
        action=agent.get_action()#先编号选择,这个就是策略（改变）
        reward=bandit.play(action)#当前action下的对应bandit奖励返回
        agent.update(action,reward)#更新Qs，为下次get_action
        total_reward+=reward
        rates.append(total_reward/(step+1))
    
    all_rates[run]=rates #更新该轮run下的胜率rates

avg_rates1=np.average(all_rates,axis=0)#求每次迭代的平均值，按照行相加平均
#


#下面采用NonstateAgent代理做action策略
runs=200
steps=1000
epsilon=0.1
alpha=0.8
all_rates=np.zeros((runs,steps))#shape为（200，1000）的数组

for run in range(runs):
    bandit=NonstateBandit()
    agent=AlphaAgent(epsilon,alpha)
    total_reward=0
    rates=[]

    for step in range(steps):
        action=agent.get_action()#当前下的编号选择，初始情况随机
        reward=bandit.play(action)#当前action下的对应bandit实际奖励返回
        agent.update(action,reward)#更新新的编号，值情况
        total_reward+=reward
        rates.append(total_reward/(step+1))
    
    all_rates[run]=rates #更新该轮run下的胜率rates

avg_rates2=np.average(all_rates,axis=0)#求每次迭代的平均值，按照行相加平均
#
plt.plot(avg_rates1, label="Sample average")
plt.plot(avg_rates2, label=f"Fixed alpha={alpha}")

plt.ylabel("Average reward")
plt.xlabel("Steps")
plt.legend()
plt.show()

