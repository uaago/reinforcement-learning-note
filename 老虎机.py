import numpy as np
import matplotlib.pyplot as plt

class bernulliBandit():
    def __init__(self,k):
        self.probs=np.random.uniform(size=k)#随机生成k个0-1的数，作为拉动没跟拉杆的获奖概率
        self.best_id=np.argmax(self.probs)
        self.best_prob=self.probs[self.best_id]
        self.k=k
    def step(self,k):
        if np.random.rand()<self.probs[k]:#如果随机数小于该拉杆的获奖概率，则奖励为1，否则为0
            return 1
        else:
            return 0
        
np.random.seed(0)
k=10
bandit1=bernulliBandit(k)
print("最优拉杆的位置是%d号,获奖概率为:%.4f" % (bandit1.best_id,bandit1.best_prob))
'''
print("随机生成了一个%d臂伯努利老虎机"%k)
print("每个拉杆的获奖概率分别为：",bandit1.probs)
print("最优拉杆的位置是%d号,获奖概率为:%.4f" % (bandit1.best_id,bandit1.best_prob))
print(type(bandit1.probs))
print(type(bandit1.best_id))
'''
class Solver():
    def __init__(self,bandit):
        self.bandit=bandit
        self.counts=np.zeros(self.bandit.k)#每个拉杆被拉动的次数
        self.regret=0#当前步的累计懊悔值
        self.actions=[]#维护一个列表，记录每一步动作
        self.regrets=[]#维护一个列表，记录每一步的累计懊悔值
    def update_regret(self,k):
        #计算累积懊悔并保存，k为本次动作选择的拉杆编号
        self.regret+=self.bandit.best_prob-self.bandit.probs[k]
        self.regrets.append(self.regret)
    def run_one_step(self):

        #执行一步动作,由每个具体的策略实现
        raise NotImplementedError
    def run(self,num_steps):
        #运行一定次数，num_steps为总运行次数
        for  _ in range(num_steps):
            k=self.run_one_step()
            self.counts[k]+=1
            self.actions.append(k)
            self.update_regret(k)
    #以上求解器中的run_one_step方法是会返回一个当前的选择k，然后函数
    # run更新m每个拉杆各自的计数counts和每一步的动作以懊悔值

class EpsilonGreeedy(Solver):
    #epsilon-greedy算法,继承Solver类
    def __init__(self,bandit,epsilon=0.1,init_prob=1.0):
        super().__init__(bandit)#等价于Solver.__init__(self,bandit)
        self.epsilon=epsilon
        #初始化拉动所有拉杆的期望奖励估值
        self.estimates=np.array([init_prob]*self.bandit.k)#因为初始化了父类，此时的self也拥有了bandit属性，所以可以直接访问self.bandit.k

    def run_one_step(self):
        if np.random.rand()<self.epsilon:
            k=np.random.choice(self.bandit.k)
            #或者 k=np.random.randinit(self.bandit.k) 
        else:#以1-epslon的概率选择当前期望奖励估值最高的拉杆
            k=np.argmax(self.estimates)
        r=self.bandit.step(k)#执行动作，获得奖励
        #更新期望奖励估值
        self.estimates[k]+=1./(self.counts[k]+1)*(r-self.estimates[k])#增量式更新期望奖励估值
        return k#注意这个r-self.estimates[k]可能为负，相当于不断逼近更真实的估计值（对k号来说）

def plot_results(solvers,solver_names):#用来画多个解释器
    for idx,solver in enumerate(solvers):
        time_list=range(len(solver.regrets))
        plt.plot(time_list,solver.regrets,label=solver_names[idx])

    plt.xlabel('time step')
    plt.ylabel('culative regrets')
    plt.legend()
    plt.show()

ep_greedy_solver=EpsilonGreeedy(bandit1,epsilon=0.01)
ep_greedy_solver.run(5000)
print("total regrets " , ep_greedy_solver.regret)
print(ep_greedy_solver.actions)
plot_results([ep_greedy_solver],['ep_greedy'])

'''


np.random.seed(0)
epsilons=[1e-4,0.01,0.1,0.25,0.5]
epsilon_greedy_solver_list=[EpsilonGreeedy(bandit1,epsilon=e) for e in epsilons]
names=["epsilon={}".format(e) for e in epsilons]
for solver in epsilon_greedy_solver_list:
    solver.run(5000)

plot_results(epsilon_greedy_solver_list,names)
'''
#画出不同epsilon下的结果

class DecayingEpsilonGreeedy(Solver):
    #epsilon-greedy算法,继承Solver类
    def __init__(self,bandit,init_prob=1.0):
        super().__init__(bandit)#等价于Solver.__init__(self,bandit)
        #self.epsilon=epsilon
        #初始化拉动所有拉杆的期望奖励估值
        self.estimates=np.array([init_prob]*self.bandit.k)#因为初始化了父类，此时的self也拥有了bandit属性，所以可以直接访问self.bandit.k
        self.total_count=0
    def run_one_step(self):
        self.total_count+=1
        self.epsilon=1./self.total_count    

        if np.random.rand()<self.epsilon:
            k=np.random.choice(self.bandit.k)
            #或者 k=np.random.randinit(self.bandit.k) 
        else:#以1-epslon的概率选择当前期望奖励估值最高的拉杆
            k=np.argmax(self.estimates)
        r=self.bandit.step(k)#执行动作，获得奖励
        #更新期望奖励估值
        self.estimates[k]+=1./(self.counts[k]+1)*(r-self.estimates[k])#增量式更新期望奖励估值
        return k#注意这个r-self.estimates[k]可能为负，相当于不断逼近更真实的估计值（对k号来说）


Decaying_ep_greedy_solver=DecayingEpsilonGreeedy(bandit1)
Decaying_ep_greedy_solver.run(5000)
print("total regrets " , Decaying_ep_greedy_solver.regret)
print(Decaying_ep_greedy_solver.actions)
plot_results([Decaying_ep_greedy_solver],['Decaying_ep_greedy'])

class UCB(Solver):
    def ___init__(self,bandit,init_prob=1.0):
        super().__init__(bandit)
        self.estimates=np.array([init_prob]*self.bandit.k)#一个一维的numpy向量
        self.total_count=0
        self.c=1.0
    def run_one_step(self):
        self.total_count+=1
       #定义当前每号臂的ucb
        ucb=self.estimates+self.c*np.sqrt(np.log(self.total_count)/(2*(self.counts+1)))
        k=np.argmax(ucb)
        r=self.bandit.step(k)
        self.estimates[k]+=1./(self.counts[k]+1)*(r-self.estimates[k])#增量式更新期望奖励估值
        return k#注意这个r-self.estimates[k]可能为负，相当于不断逼近更真实的估计值（对k号来说）

upper_confidence_solver=DecayingEpsilonGreeedy(bandit1)
upper_confidence_solver.run(5000)
print("total regrets " , upper_confidence_solver.regret)
print(upper_confidence_solver.actions)
plot_results([upper_confidence_solver],['upper_confidence'])
