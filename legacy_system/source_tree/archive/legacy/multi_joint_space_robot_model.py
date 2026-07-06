import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils_plot import save_figure, get_save_dir
import os

class MultiJointSpaceRobot:
    """
    多关节绳驱空间机械臂模型 (5 根连杆，2 个独立自由度)
    - Link 1: 固定基座连杆
    - Link 2: 第一级绳驱十字架 (横轴)
    - Link 3: 第一级主活动连杆
    - Link 4: 第二级绳驱十字架 (横轴)
    - Link 5: 第二级主活动连杆
    
    物理耦合约束:
    - Joint 1 (Link1-Link2) 与 Joint 2 (Link2-Link3) 角度相等，统称 qa
    - Joint 3 (Link3-Link4) 与 Joint 4 (Link4-Link5) 角度相等，统称 qb
    """
    def __init__(self):
        # 1. 运动学参数 (长度 单位: 米)
        self.L1 = 2.0
        self.L2 = 0.2
        self.L3 = 2.0
        self.L4 = 0.2
        self.L5 = 2.0
        
        # 十字架 (Spreader) 长度
        self.Ls1 = 3.0 * self.L2
        self.Ls2 = 3.0 * self.L4
        
        # 2. 动力学参数 (质量 单位: kg)
        self.m2 = 1.161  
        self.m3 = 2.866
        self.m4 = 1.161
        self.m5 = 2.866
        
        self.r2 = self.L2 / 2.0
        self.r3 = self.L3 / 2.0
        self.r4 = self.L4 / 2.0
        self.r5 = self.L5 / 2.0
        
        self.I2 = (self.m2 * self.L2**2) / 12.0
        self.I3 = (self.m3 * self.L3**2) / 12.0
        self.I4 = (self.m4 * self.L4**2) / 12.0
        self.I5 = (self.m5 * self.L5**2) / 12.0

    # ==========================================
    # 模块 A: 正运动学与逆运动学
    # ==========================================
    def forward_kinematics(self, qa, qb):
        """
        计算各连杆端点的坐标
        """
        p0 = np.array([0.0, 0.0])
        p1 = p0 + np.array([self.L1, 0.0])
        
        # Link 2 绝对角度: qa
        p2 = p1 + np.array([self.L2 * np.cos(qa), self.L2 * np.sin(qa)])
        
        # Link 3 绝对角度: qa + qa = 2*qa
        p3 = p2 + np.array([self.L3 * np.cos(2*qa), self.L3 * np.sin(2*qa)])
        
        # Link 4 绝对角度: 2*qa + qb
        p4 = p3 + np.array([self.L4 * np.cos(2*qa + qb), self.L4 * np.sin(2*qa + qb)])
        
        # Link 5 绝对角度: 2*qa + qb + qb = 2*qa + 2*qb
        p5 = p4 + np.array([self.L5 * np.cos(2*qa + 2*qb), self.L5 * np.sin(2*qa + 2*qb)])
        
        return p0, p1, p2, p3, p4, p5

    def inverse_kinematics(self, target_p5, q_guess=None, max_iter=200, tol=1e-5):
        """
        数值法求解逆运动学 (Newton-Raphson / Levenberg-Marquardt)
        已知目标末端点 target_p5 = [x, y]，求解 [qa, qb]
        
        由于 2 个自由度 (qa, qb) 控制 2 个工作空间坐标 (x, y)，
        系统通常存在有限个解。我们利用空间雅可比矩阵进行迭代求解。
        """
        if q_guess is None:
            q = np.array([0.1, 0.1])  # 避开完全伸直的奇异点
        else:
            q = np.array(q_guess, dtype=float)
            
        target_p5 = np.array(target_p5, dtype=float)
        
        for i in range(max_iter):
            # 当前估计位置
            p_all = self.forward_kinematics(q[0], q[1])
            p5 = p_all[-1]
            err = target_p5 - p5
            
            # 如果误差已经足够小，收敛成功
            if np.linalg.norm(err) < tol:
                return q, True
                
            # 计算解析雅可比矩阵 J(q) (2x2)
            qa, qb = q[0], q[1]
            
            dx_dqa = -self.L2 * np.sin(qa) - 2 * self.L3 * np.sin(2*qa) \
                     - 2 * self.L4 * np.sin(2*qa + qb) - 2 * self.L5 * np.sin(2*qa + 2*qb)
            dx_dqb = -self.L4 * np.sin(2*qa + qb) - 2 * self.L5 * np.sin(2*qa + 2*qb)
            
            dy_dqa =  self.L2 * np.cos(qa) + 2 * self.L3 * np.cos(2*qa) \
                     + 2 * self.L4 * np.cos(2*qa + qb) + 2 * self.L5 * np.cos(2*qa + 2*qb)
            dy_dqb =  self.L4 * np.cos(2*qa + qb) + 2 * self.L5 * np.cos(2*qa + 2*qb)
            
            J = np.array([
                [dx_dqa, dx_dqb],
                [dy_dqa, dy_dqb]
            ])
            
            # 增量更新 (使用阻尼最小二乘法 Levenberg-Marquardt 避免奇异点崩溃)
            lambda_damp = 1e-4
            J_pinv = J.T @ np.linalg.inv(J @ J.T + lambda_damp * np.eye(2))
            delta_q = J_pinv @ err
            
            q = q + delta_q
            
        return q, False  # 超过最大迭代次数未收敛

    def get_crossbars_and_cables(self, qa, qb):
        """
        计算两个绳驱十字架及8根绳索锚点坐标
        """
        p0, p1, p2, p3, p4, p5 = self.forward_kinematics(qa, qb)
        
        # --- 第一级十字架 (Link 2 上) ---
        xc1 = (p1[0] + p2[0]) / 2.0
        yc1 = (p1[1] + p2[1]) / 2.0
        nx1 = -np.sin(qa)
        ny1 = np.cos(qa)
        ps1_top = np.array([xc1 + self.Ls1/2 * nx1, yc1 + self.Ls1/2 * ny1])
        ps1_bot = np.array([xc1 - self.Ls1/2 * nx1, yc1 - self.Ls1/2 * ny1])
        # 绳索锚点：基座(p0) 和 Link3尾端(p3)
        anchors1_start = p0
        anchors1_end = p3
        
        # --- 第二级十字架 (Link 4 上) ---
        xc2 = (p3[0] + p4[0]) / 2.0
        yc2 = (p3[1] + p4[1]) / 2.0
        nx2 = -np.sin(2*qa + qb)
        ny2 = np.cos(2*qa + qb)
        ps2_top = np.array([xc2 + self.Ls2/2 * nx2, yc2 + self.Ls2/2 * ny2])
        ps2_bot = np.array([xc2 - self.Ls2/2 * nx2, yc2 - self.Ls2/2 * ny2])
        # 绳索锚点：Link3首端(p2) 和 Link5尾端(p5)
        anchors2_start = p2
        anchors2_end = p5
        
        return (ps1_top, ps1_bot, anchors1_start, anchors1_end), (ps2_top, ps2_bot, anchors2_start, anchors2_end)

    # ==========================================
    # 模块 B: 动力学建模 (拉格朗日雅可比法)
    # ==========================================
    def get_M(self, q):
        """计算 2x2 等效质量矩阵 M(q)"""
        qa, qb = q[0], q[1]
        
        # 预计算各连杆的绝对角度
        th2 = qa
        th3 = 2*qa
        th4 = 2*qa + qb
        th5 = 2*qa + 2*qb
        
        # 速度雅可比矩阵 Jv (2x2) = d(p_com) / dq
        # Link 2
        Jv2 = np.array([
            [-self.r2 * np.sin(th2), 0.0],
            [ self.r2 * np.cos(th2), 0.0]
        ])
        # Link 3
        Jv3 = np.array([
            [-self.L2 * np.sin(th2) - 2*self.r3 * np.sin(th3), 0.0],
            [ self.L2 * np.cos(th2) + 2*self.r3 * np.cos(th3), 0.0]
        ])
        # Link 4
        Jv4 = np.array([
            [-self.L2 * np.sin(th2) - 2*self.L3 * np.sin(th3) - 2*self.r4 * np.sin(th4), -self.r4 * np.sin(th4)],
            [ self.L2 * np.cos(th2) + 2*self.L3 * np.cos(th3) + 2*self.r4 * np.cos(th4),  self.r4 * np.cos(th4)]
        ])
        # Link 5
        Jv5 = np.array([
            [-self.L2 * np.sin(th2) - 2*self.L3 * np.sin(th3) - 2*self.L4 * np.sin(th4) - 2*self.r5 * np.sin(th5), 
             -self.L4 * np.sin(th4) - 2*self.r5 * np.sin(th5)],
            [ self.L2 * np.cos(th2) + 2*self.L3 * np.cos(th3) + 2*self.L4 * np.cos(th4) + 2*self.r5 * np.cos(th5),  
              self.L4 * np.cos(th4) + 2*self.r5 * np.cos(th5)]
        ])
        
        # 角速度雅可比矩阵 Jw (1x2)
        Jw2 = np.array([[1.0, 0.0]])
        Jw3 = np.array([[2.0, 0.0]])
        Jw4 = np.array([[2.0, 1.0]])
        Jw5 = np.array([[2.0, 2.0]])
        
        # 组装质量矩阵
        M = self.m2 * (Jv2.T @ Jv2) + self.I2 * (Jw2.T @ Jw2) + \
            self.m3 * (Jv3.T @ Jv3) + self.I3 * (Jw3.T @ Jw3) + \
            self.m4 * (Jv4.T @ Jv4) + self.I4 * (Jw4.T @ Jw4) + \
            self.m5 * (Jv5.T @ Jv5) + self.I5 * (Jw5.T @ Jw5)
            
        return M

    def get_C(self, q, dq):
        """利用克里斯托费尔符号数值计算科氏力矩阵 C(q, dq)"""
        eps = 1e-5
        C = np.zeros((2, 2))
        for k in range(2):
            for j in range(2):
                c_kj = 0.0
                for i in range(2):
                    # 偏导数 dM_kj / dq_i
                    q_p = q.copy(); q_p[i] += eps
                    q_m = q.copy(); q_m[i] -= eps
                    dMkj_dqi = (self.get_M(q_p)[k, j] - self.get_M(q_m)[k, j]) / (2*eps)
                    
                    # 偏导数 dM_ki / dq_j
                    q_p = q.copy(); q_p[j] += eps
                    q_m = q.copy(); q_m[j] -= eps
                    dMki_dqj = (self.get_M(q_p)[k, i] - self.get_M(q_m)[k, i]) / (2*eps)
                    
                    # 偏导数 dM_ij / dq_k
                    q_p = q.copy(); q_p[k] += eps
                    q_m = q.copy(); q_m[k] -= eps
                    dMij_dqk = (self.get_M(q_p)[i, j] - self.get_M(q_m)[i, j]) / (2*eps)
                    
                    c_kj += 0.5 * (dMkj_dqi + dMki_dqj - dMij_dqk) * dq[i]
                C[k, j] = c_kj
        return C

    def step_coupled(self, q, dq, tau, dt=0.01):
        """
        物理步进
        q = [qa, qb], dq = [dqa, dqb], tau = [tau_a, tau_b]
        """
        q = np.array(q, dtype=float)
        dq = np.array(dq, dtype=float)
        tau = np.array(tau, dtype=float)
        
        M = self.get_M(q)
        C = self.get_C(q, dq)
        
        # M * ddq + C * dq = tau
        ddq = np.linalg.inv(M) @ (tau - C @ dq)
        
        dq_next = dq + ddq * dt
        q_next = q + dq_next * dt
        
        # 物理硬限位约束 [-90度, 90度]
        q_limit = np.pi / 2.0
        for i in range(2):
            if q_next[i] > q_limit:
                q_next[i] = q_limit
                dq_next[i] = 0.0
                print(f"⚠️ 警告: 关节 {i+1} 超出正向限位！")
            elif q_next[i] < -q_limit:
                q_next[i] = -q_limit
                dq_next[i] = 0.0
                print(f"⚠️ 警告: 关节 {i+1} 超出负向限位！")
                
        return q_next, dq_next


# ==========================================
# 测试与可视化模块
# ==========================================
if __name__ == "__main__":
    robot = MultiJointSpaceRobot()
    
    print("开始多关节绳驱太空机械臂仿真测试...")
    q = np.array([0.0, 0.0])
    dq = np.array([0.0, 0.0])
    dt = 0.05
    steps = 150
    
    history_q = []
    history_x5 = []
    history_y5 = []
    
    for step in range(steps):
        # 施加复合摇摆力矩，让两个关节产生复杂的非线性耦合动态
        tau = np.array([
            5.0 * np.sin(step * 0.1), 
            2.0 * np.cos(step * 0.15)
        ])
        
        history_q.append(q.copy())
        
        # 记录末端点 p5
        p_all = robot.forward_kinematics(q[0], q[1])
        p5 = p_all[-1]
        history_x5.append(p5[0])
        history_y5.append(p5[1])
        
        q, dq = robot.step_coupled(q, dq, tau, dt)

    # ----------------------------------------------------
    # 生成炫酷的 GIF 动图
    # ----------------------------------------------------
    print("正在渲染多关节绳驱机构动画...")
    fig_anim = plt.figure(figsize=(12, 6))
    ax = fig_anim.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-1.0, 9.0), ylim=(-4.0, 4.0))
    ax.grid(True)
    ax.set_title("Multi-Joint Cable-Driven Space Robot (5 Links, 2 DOFs)")
    
    # 连杆
    line_links, = ax.plot([], [], color='#FF8C00', linestyle='-', linewidth=8, label='Main Links')
    line_s1, = ax.plot([], [], color="#0026FFFF", linestyle='-', linewidth=4, label='Spreaders')
    line_s2, = ax.plot([], [], color="#0026FFFF", linestyle='-', linewidth=4)
    
    # 关节与末端
    point_joints, = ax.plot([], [], 'ro', markersize=5, zorder=4, label='Joints')
    point_end, = ax.plot([], [], 'y*', markersize=14, zorder=6)
    
    # 绳索
    cable_style = {'color': "#000000", 'linestyle': '--', 'linewidth': 1.0}
    cables1 = [ax.plot([], [], **cable_style)[0] for _ in range(4)]
    cables2 = [ax.plot([], [], **cable_style)[0] for _ in range(4)]
    
    # 铰链
    hinges_cable, = ax.plot([], [], 'o', markersize=4, markerfacecolor='white', markeredgecolor='black', zorder=5)
    
    # 拖尾
    line_traj, = ax.plot([], [], 'm-', alpha=0.4, linewidth=2, label='End Effector Path')
    
    ax.legend(loc='upper right', fontsize=9)

    def init():
        return (line_links, line_s1, line_s2, point_joints, point_end, hinges_cable, line_traj, *cables1, *cables2)

    def animate(i):
        qa, qb = history_q[i]
        p_all = robot.forward_kinematics(qa, qb)
        p0, p1, p2, p3, p4, p5 = p_all
        
        # 绘制主干
        xs = [p[0] for p in p_all]
        ys = [p[1] for p in p_all]
        line_links.set_data(xs, ys)
        
        # 关节
        point_joints.set_data(xs[1:-1], ys[1:-1])
        point_end.set_data([p5[0]], [p5[1]])
        
        # 十字架与绳索
        (ps1_t, ps1_b, anc1_s, anc1_e), (ps2_t, ps2_b, anc2_s, anc2_e) = robot.get_crossbars_and_cables(qa, qb)
        
        line_s1.set_data([ps1_b[0], ps1_t[0]], [ps1_b[1], ps1_t[1]])
        line_s2.set_data([ps2_b[0], ps2_t[0]], [ps2_b[1], ps2_t[1]])
        
        # 绳索组 1
        cables1[0].set_data([ps1_t[0], anc1_s[0]], [ps1_t[1], anc1_s[1]]) # cable11
        cables1[1].set_data([ps1_b[0], anc1_s[0]], [ps1_b[1], anc1_s[1]]) # cable12
        cables1[2].set_data([ps1_t[0], anc1_e[0]], [ps1_t[1], anc1_e[1]]) # cable13
        cables1[3].set_data([ps1_b[0], anc1_e[0]], [ps1_b[1], anc1_e[1]]) # cable14
        
        # 绳索组 2
        cables2[0].set_data([ps2_t[0], anc2_s[0]], [ps2_t[1], anc2_s[1]]) # cable21
        cables2[1].set_data([ps2_b[0], anc2_s[0]], [ps2_b[1], anc2_s[1]]) # cable22
        cables2[2].set_data([ps2_t[0], anc2_e[0]], [ps2_t[1], anc2_e[1]]) # cable23
        cables2[3].set_data([ps2_b[0], anc2_e[0]], [ps2_b[1], anc2_e[1]]) # cable24
        
        # 铰链点
        hx = [anc1_s[0], anc1_e[0], ps1_t[0], ps1_b[0], anc2_s[0], anc2_e[0], ps2_t[0], ps2_b[0]]
        hy = [anc1_s[1], anc1_e[1], ps1_t[1], ps1_b[1], anc2_s[1], anc2_e[1], ps2_t[1], ps2_b[1]]
        hinges_cable.set_data(hx, hy)
        
        # 轨迹
        line_traj.set_data(history_x5[:i+1], history_y5[:i+1])
        
        return (line_links, line_s1, line_s2, point_joints, point_end, hinges_cable, line_traj, *cables1, *cables2)

    ani = animation.FuncAnimation(fig_anim, animate, frames=steps, interval=dt*1000, blit=True, init_func=init)
    
    save_dir = get_save_dir()
    gif_path = os.path.join(save_dir, "multi_joint_space_robot.gif")
    ani.save(gif_path, writer='pillow', fps=int(1/dt))
    print(f"[动图保存] 成功！GIF 动画已保存至: {gif_path}")
    
    # 画静态轨迹图
    plt.figure(figsize=(6, 6))
    plt.plot(history_x5, history_y5, 'm-', linewidth=2, label="End Effector Path")
    plt.plot(history_x5[0], history_y5[0], 'go', markersize=8, label="Start")
    plt.plot(history_x5[-1], history_y5[-1], 'rs', markersize=8, label="End")
    plt.title("Multi-Joint End Effector Trajectory")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box') 
    save_figure(fig_name="trajectory")
    
    # plt.show()
