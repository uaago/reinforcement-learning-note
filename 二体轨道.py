import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

mu=3.986004418e14  # 地球引力常数 m^3/s^2

def two_body(t,state):
    r=state[0:3]
    v=state[3:6]
    r_norm=np.linalg.norm(r)
    a=-mu/(r_norm**3)*r
    return np.concatenate([v,a])
#进行积分并且画图，在地球惯性坐标系下
R=6371e3
h=400e3
r0=np.array([R+h,0,0])
v_circle=np.sqrt(mu/np.linalg.norm(r0))
v0=np.array([0,v_circle,0])
T=2*np.pi*np.sqrt(np.linalg.norm(r0)**3/mu)
state0=np.concatenate((r0,v0))
sol=solve_ivp(two_body,t_span=(0,T),y0=state0,method='RK45',rtol=1e-9,atol=1e-12,dense_output=True
)

#画图
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.plot(sol.y[0],sol.y[1],sol.y[2])
plt.show()