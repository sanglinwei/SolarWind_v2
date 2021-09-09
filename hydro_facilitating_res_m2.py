#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This file works on matrices given different uncertainty"""
"""This file works on the fixed the ratio and find the change rules under the different ratios"""

__author__ = "Linwei Sang"
__email__ = "sanglinwei@gmail.com"

import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl
import pandas as pd
import cvxpy as cp
import numpy as np
from matplotlib import cm
from tqdm import tqdm

matplotlib.style.use('default')
matplotlib.rc('font', size=12)
# plt.rcParams['axes.unicode_minus']=False #
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
font = mpl.font_manager.FontProperties(fname='./SimSun.ttc')

# stochastic optimization
if __name__ == '__main__':
    # 风光相关数据
    typical_day = np.load('./typicaldaySC.npy')
    select_day = typical_day[24 * 0:24 * 0 + 24, ]
    # typical_day = typical_day[24*1:,]
    for idx in [2, 3, 5, 6, 7, 8]:
        select_day = np.concatenate((select_day, typical_day[24 * idx:24 * idx + 24, ]), axis=0)

    typical_day = select_day
    ratio_sw = 0.5  # 光伏与风电的接入比例
    T = typical_day.shape[0]
    solar = typical_day[:, 0][0:T]  # 典型日光伏出力
    wind = typical_day[:, 1][0:T]  # 典型日风电出力
    demand = typical_day[:, 2][0:T] - 0.2  # 典型日负荷出力
    demand = demand / demand.max()

    # 水电相关数据
    inflow_month_pd = pd.read_csv('./inputdata/inflow.csv')
    inflow_month_np = inflow_month_pd.iloc[:, 1:].to_numpy().T
    station_month_np = inflow_month_np[:12, 1] / inflow_month_np[:12, 1].max()
    # 生成小时级的p.u.值数据
    station_month_ls = station_month_np.tolist()
    month_day = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    station_hour_ls = []
    for flow, day in zip(station_month_ls, month_day):
        station_hour_ls += [flow] * 24 * day
    station_hour_np = np.array(station_hour_ls)
    Natural_inflow = 15e4  # 单位是m^3/h
    eta = 6.197
    h0 = 0.82678
    alpha = 0.000042
    V_max = 240e4
    V_min = 100e4
    Ramp = 60
    P_h_min = 7
    P_h_max = 120
    # 小时级的自然水流
    inflow_hour_np = Natural_inflow * station_hour_np

    # 随机优化优化模型

    # 水电站基本信息
    Q_min = 0
    Q_max = 20e4
    # 小时级的水电信息
    Natural_inflow = 15e4  # 单位是m^3/h
    eta = 6.197
    h0 = 0.82678
    alpha = 0.000042
    V_max = 240e4
    V_min = 100e4
    Ramp = 60
    P_h_min = 0  # 7
    # P_h_max = 120
    P_h_max = cp.Variable(nonneg=True)
    # 小时级的自然水流
    inflow_hour_np = Natural_inflow * station_hour_np

    sigma = 0.2  # 不确定度
    N_scenario = 5  # 场景数目
    cap_dic = {}  # 存储不同场景下的消纳容量
    thermal_dic = {}  # 存储不同场景下的火电需求
    for n_scenario in range(N_scenario):
        T = 24  # 计算周期
        # 优化过程中的库容
        V_h = cp.Variable(T, nonneg=True)
        # 初始库容
        V_h_init = cp.Variable(1, nonneg=True)
        # discharge capacity (m3)
        Q_h = cp.Variable(T, nonneg=True)
        # discharge power (MW)
        P_h = cp.Variable(T, nonneg=True)

        # 水电站基本模型 非线性项处理
        M = 4
        N = 3
        Ph_np = np.zeros(shape=(M, N))
        # 水流量的分段
        qz_np = np.linspace(start=Q_min, stop=Q_max, num=M)
        # 水库体积的分段
        vy_np = np.linspace(start=V_min, stop=V_max, num=N)
        for m in range(M):
            for n in range(N):
                Ph_np[m, n] = (eta * qz_np[m] * (h0 + alpha * vy_np[n])) * 1e-6
        lam_dic = {}
        L_dic = {}
        R_dic = {}
        for t in range(T):
            lam_dic[t] = cp.Variable(shape=(M, N), nonneg=True)
            L_dic[t] = cp.Variable(shape=(M, N), boolean=True)
            R_dic[t] = cp.Variable(shape=(M, N), boolean=True)
        constr = []
        for t in range(T):
            constr += [cp.sum(cp.sum(lam_dic[t])) == 1]
            constr += [Q_h[t] == cp.sum(lam_dic[t].T @ qz_np)]
            constr += [V_h[t] == cp.sum(lam_dic[t] @ vy_np)]
            constr += [P_h[t] == cp.sum(cp.sum(cp.multiply(Ph_np, lam_dic[t])))]
            constr += [cp.sum(cp.sum(R_dic[t] + L_dic[t])) == 1]
            constr += [L_dic[t][:, 0] == 0]
            constr += [L_dic[t][M - 1, :] == 0]
            constr += [R_dic[t][:, N - 1] == 0]
            constr += [R_dic[t][0, :] == 0]
            for m in range(M):
                for n in range(N):
                    rh = L_dic[t][m, n] + R_dic[t][m, n]
                    if m != 0:
                        rh += L_dic[t][m - 1, n]
                    if n != 0:
                        rh += R_dic[t][m, n - 1]
                    if m != M - 1:
                        rh += R_dic[t][m + 1, n]
                    if n != N - 1:
                        rh += L_dic[t][m, n + 1]
                    constr += [lam_dic[t][m, n] <= rh]
                    rh = 0
        # 水电站范围约束
        for t in range(T):
            constr += [Q_h[t] <= Q_max, Q_h[t] >= Q_min]
            constr += [V_h[t] <= V_max, V_h[t] >= V_min]
            constr += [P_h[t] <= P_h_max, P_h[t] >= P_h_min]
        # 水电站容量耦合关系
        for t in range(1, T):
            constr += [V_h[t] == V_h[t - 1] + inflow_hour_np[t] - Q_h[t]]
        constr += [V_h[0] == V_h_init + inflow_hour_np[0] - Q_h[0]]
        constr += [V_h_init <= V_max, V_h_init >= V_min]

        # 抽水蓄能建模
        C_ps = cp.Parameter()
        P_psmax = cp.Parameter()

        E = cp.Variable(T, nonneg=True)
        p_ps = cp.Variable(T)
        c_0 = 0.5
        c_ps_min = 0
        c_ps_max = 1

        cost_ps = 1e2  # 抽水蓄能电站成本
        Cost_cps = 1  # 抽水蓄能电站的装机容量成本
        Cost_pps = 1  # 抽水蓄能电站的库容量成本

        # 抽水蓄能建模
        for t in range(1, T):
            constr += [E[t] == E[t - 1] - p_ps[t]]
            if (t + 1) % 24 == 0:
                constr += [cp.abs(E[t] - E[t - 24]) <= 0.1 * C_ps]
        constr += [E[0] == c_0 * C_ps - p_ps[0]]
        constr += [c_ps_min * C_ps <= E]
        constr += [c_ps_max * C_ps >= E]
        constr += [p_ps <= P_psmax]
        constr += [p_ps >= - P_psmax]

        # 火电站
        p_g = cp.Variable(T, nonneg=True)
        c_g_min = 0.2
        c_g_max = 1
        # 火电容量
        C_g = cp.Variable(nonneg=True)
        constr += [p_g >= c_g_min * C_g]
        constr += [p_g <= c_g_max * C_g]

        # 风光建模
        ratio = cp.Parameter()
        # 风光接入容量
        C_sw = cp.Variable(nonneg=True)
        # C_sw = cp.Parameter(nonneg=True)
        # 风光打捆
        P_SW = (solar[:T] * ratio + wind[:T] * (1 - ratio)) * C_sw
        # 考虑随机性
        k_random = np.random.uniform(-0.8, 0.8, T)
        P_SW = P_SW + sigma * k_random * P_SW

        p_sw = cp.Variable(T, nonneg=True)
        # 考虑弃风
        drop_sw = cp.Parameter()
        constr += [p_sw >= 0, p_sw <= P_SW]
        constr += [cp.sum(P_SW - p_sw) <= drop_sw * cp.sum(P_SW)]
        # 负荷建模
        C_d = cp.Parameter()
        P_d = C_d * demand[:T]
        # 功率平衡 P_h 120 WM
        constr += [p_sw + p_ps + P_h + p_g == P_d]
        # 最小化抽水蓄能成本
        # obj = cp.Minimize(Cost_cps * C_ps + Cost_pps * P_psmax)
        # 最大化消纳风光的成本
        obj = cp.Maximize(C_sw - 2000 * C_g)
        problem = cp.Problem(obj, constr)

        ratio.value = 1
        C_d.value = 200
        P_h_max.value = 100
        P_psmax.value = 0
        C_ps.value = 300000
        drop_sw.value = 0.1
        problem.solve(solver=cp.GUROBI)
        print('消纳风光的容量{}'.format(C_sw.value))
        print('单位水电支持多少风光{}'.format(C_sw.value / (P_h_max.value + P_psmax.value)))

        # 抽蓄占比
        ratio_hp_np = np.linspace(0, 1, 10 + 1)
        # 光占比
        ratio_sw_np = np.linspace(0, 1, 10 + 1)
        # 消纳风光的容量
        cap_sw_mat = np.zeros((ratio_sw_np.shape[0], ratio_hp_np.shape[0]))
        # 单位风光支持多少水电
        ratio_sw_mat = np.zeros((ratio_sw_np.shape[0], ratio_hp_np.shape[0]))
        # 火电需求
        thermal_mat = np.zeros((ratio_sw_np.shape[0], ratio_hp_np.shape[0]))

        for x_idx, v in enumerate(tqdm(ratio_hp_np)):
            for y_idx, r in enumerate(ratio_sw_np):
                C_d.value = 200
                C_ps.value = 30000
                drop_sw.value = 0.1
                ratio.value = r
                P_h_max.value = 100 * (1 - v)
                P_psmax.value = 100 * v
                problem.solve(solver=cp.GUROBI)
                cap_sw_mat[y_idx, x_idx] = C_sw.value
                ratio_sw_mat[y_idx, x_idx] = C_sw.value / 100
                thermal_mat[y_idx, x_idx] = C_g.value
        # 保存不同场景下的优化结果
        cap_dic[n_scenario] = cap_sw_mat
        thermal_dic[n_scenario] = thermal_mat

    cap_cat = np.dstack([cap_dic[i] for i in range(N_scenario)])
    thermal_cat = np.dstack([thermal_dic[i] for i in range(N_scenario)])
    # 求鲁棒的结果（最恶劣场景）
    cap_rob = np.min(cap_cat, axis=2)
    thermal_rob = np.min(thermal_cat, axis=2)
    # 求随机结果（平均场景）
    cap_mean = np.mean(cap_cat, axis=2)
    thermal_mean = np.mean(thermal_cat, axis=2)

    # 绘制消纳容量
    X, Y = np.meshgrid(ratio_hp_np, ratio_sw_np)
    plt.rc('font', family='Times New Roman', style='normal', size=13)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    surf = ax.plot_surface(X, Y, cap_mean[:, :], cmap=cm.coolwarm, linewidth=0, antialiased=False)
    surf = ax.plot_surface(X, Y, cap_rob[:, :], cmap=cm.coolwarm, linewidth=0, antialiased=False)
    cset = ax.contourf(X, Y, cap_rob[:, :], zdir='z', offset=np.min(cap_rob[:, :]), cmap=cm.coolwarm)
    ax.set_xlabel('抽蓄占比', fontproperties=font)
    ax.set_ylabel('光占比', fontproperties=font, rotation=50)
    ax.set_zlabel('新能源消纳容量/MW', fontproperties=font)
    plt.margins(x=0)
    plt.margins(y=0)
    plt.grid()
    plt.colorbar(surf, ax=[ax], location='left', shrink=0.7, aspect=10, pad=0)
    plt.savefig('./figs/消纳容量m2.png', dpi=900, transparent=True, pad_inches=0)
    plt.show()

    # 绘制对火电的需求
    X, Y = np.meshgrid(ratio_hp_np, ratio_sw_np)
    plt.rc('font', family='Times New Roman', style='normal', size=13)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    surf = ax.plot_surface(X, Y, thermal_rob[:, :], cmap=cm.coolwarm, linewidth=0, antialiased=False)
    surf = ax.plot_surface(X, Y, thermal_mean[:, :], cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel('抽蓄占比', fontproperties=font)
    ax.set_ylabel('光占比', fontproperties=font, rotation=50)
    ax.set_zlabel('火电装机/MW', fontproperties=font)
    plt.margins(x=0)
    plt.margins(y=0)
    plt.grid()
    plt.colorbar(surf, ax=[ax], location='left', shrink=0.7, aspect=10, pad=0)
    plt.savefig('./figs/火电调节能力m2.png', dpi=900, transparent=True, pad_inches=0)
    plt.show()
