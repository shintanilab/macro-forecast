# プログラム2.1
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from statsmodels.tsa import filters as filt
from statsmodels.tsa.stattools import adfuller



#####
from matplotlib import pyplot as plt
from matplotlib.pylab import rcParams
rcParams["figure.figsize"] = 15, 6
import japanize_matplotlib



# プログラム2.2
# データの読み込み
gdp = pd.read_excel("data_2.xlsx", sheet_name="GDP暦年(内閣府)", header=[0])
ngdp, rgdp, gdpdef = gdp.iloc[:, 1], gdp.iloc[:, 2], gdp.iloc[:, 3]
gdpdef_cal = ngdp / rgdp * 100
# データの出力
print(gdp)



#####
# グラフの描画
fig = plt.figure(figsize=(15, 13))
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
dti = pd.date_range("1994", periods=len(gdp), freq="YS")

ax1.set_title("(a)名目GDPと実質GDP", fontsize=24)
ax1.plot(dti, ngdp / 1000, label="名目GDP", color="black")
ax1.plot(dti, rgdp / 1000, label="実質GDP", color="black", linestyle = "dotted", marker="D")
ax1.legend(loc="lower right", fontsize=16)
ax1.tick_params(labelsize=16)
ax1.set_ylabel("兆円", fontsize=16)
ax2.set_title("(b)GDPデフレータ", fontsize=24)
ax2.plot(dti, gdpdef, label="公表値", color="black")
ax2.plot(dti, gdpdef_cal, label="逆算値", color="black", linestyle = "dotted", marker="D")
ax2.legend(loc="lower left", fontsize=16)
ax2.tick_params(labelsize=16)
ax2.set_ylabel("2015年=100", fontsize=16)



# プログラム2.3
# 変化率と対数近似の変化率
rch = (rgdp[1:].to_numpy() / rgdp[:-1].to_numpy() - 1) * 100
lch = (np.log(rgdp[1:].to_numpy()) - np.log(rgdp[:-1].to_numpy())) * 100



#####
# グラフの描画
plt.title("実質GDP成長率", fontsize=24)
plt.plot(dti[1:], rch, label="変化率", color="black")
plt.plot(dti[1:], lch, label="対数近似の変化率", color="black", linestyle="dotted", marker="D")
plt.legend(loc="lower left", fontsize=16)
plt.tick_params(labelsize=16)
plt.ylabel("％", fontsize=16)
plt.savefig("F2-2.png", dpi=400, bbox_inches="tight")



# プログラム2.4
# 季節調整と移動平均
# データの読み込み
qgdp = pd.read_excel("data_2.xlsx", sheet_name="GDP四半期原計数(内閣府)", header=[0])
qdti = pd.date_range("1994-01-01", periods=len(qgdp), freq="QS")
rqgdp = qgdp.iloc[:, 2]

# 前年同期比
qrch = (rqgdp[4:].to_numpy() / rqgdp[:-4].to_numpy() - 1) * 100
qlch = (np.log(rqgdp[4:].to_numpy()) - np.log(rqgdp[:-4].to_numpy())) * 100

# 後方移動平均・中心化移動平均
qqlch = (np.log(rqgdp[1:].to_numpy()) - np.log(rqgdp[:-1].to_numpy())) * 400
BMA_qqlch = (pd.DataFrame(qqlch)).rolling(window=4, center=False).mean()
CMA_qqlch = (
    (pd.DataFrame(qqlch)).rolling(window=4, center=False).mean().shift(-1)
    + (pd.DataFrame(qqlch)).rolling(window=4, center=False).mean().shift(-2)
) / 2



#####
# グラフの描画
fig = plt.figure(figsize=(15, 20))
ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3)
ax1.set_title("(a)四半期実質GDP", fontsize=24)
ax1.plot(qdti, rqgdp / 1000, label="原系列", color="black")
ax1.legend(loc="lower right", fontsize=16)
ax1.tick_params(labelsize=16)
ax1.set_ylabel("兆円", fontsize=16)
# 前年同期比
ax2.set_title("(b)四半期実質GDPの前年比", fontsize=24)
ax2.plot(qdti[4:], qrch, label="変化率", color="black")
ax2.plot(qdti[4:], qlch, label="対数近似の変化率", color="black", linestyle="dotted", marker="D")
ax2.legend(loc="lower left", fontsize=16)
ax2.tick_params(labelsize=16)
ax2.set_ylabel("％", fontsize=16)
# 後方4四半期移動平均・中心4四半期移動平均
ax3.set_title("(c)四半期実質GDPの成長率", fontsize=24)
ax3.plot(qdti[4:], qlch, label="前年同期との対数近似の変化率", color="black")
ax3.plot(
    qdti[1:],
    (np.log(rqgdp[1:].to_numpy()) - np.log(rqgdp[:-1].to_numpy())) * 100,
    label="前期との対数近似の変化率",
    color="black",
    linestyle="dotted",
)
ax3.plot(qdti[1:], BMA_qqlch.to_numpy(), label="前期との対数近似の変化率の後方移動平均", color="black", linestyle="dashed", marker="D")
ax3.plot(qdti[1:], CMA_qqlch.to_numpy(), label="前期との対数近似の変化率の中心化移動平均", color="black", linestyle="dashdot")
ax3.legend(loc="lower left", fontsize=16)
ax3.tick_params(labelsize=16)
ax3.set_ylabel("％", fontsize=16)



# プログラム2.5
# データの読み込み
qsagdp = pd.read_excel("data_2.xlsx", sheet_name="GDP四半期季節調整済(内閣府)", header=[0])
rqsagdp = qsagdp.iloc[:, 2]
gap = qsagdp.iloc[:, 4]

# HPフィルタ(調整パラメータ=1600)を計算
hp = filt.hp_filter.hpfilter(np.log(rqsagdp), lamb=1600)
hp_cycle, hp_trend = pd.DataFrame(hp[0]), pd.DataFrame(hp[1])



#####
# グラフの描画
fig = plt.figure(figsize=(15, 13))
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
ax1.set_title("(a)実質GDPとトレンド成分", fontsize=24)
ax1.plot(qdti, (np.exp(hp_trend) / 1000).to_numpy(), label="HPフィルタによるトレンド成分", color="black")
ax1.plot(qdti, rqsagdp / 1000, label="実質GDP", color="black", linestyle="dashed")
ax1.legend(loc="lower right", fontsize=16)
ax1.tick_params(labelsize=16)
ax1.set_ylabel("兆円", fontsize=16)
ax2.set_title("(b)実質GDPの循環成分とGDPギャップ", fontsize=24)
ax2.plot(qdti, (hp_cycle.to_numpy()) * 100, label="HPフィルタによる循環成分", color="black")
ax2.plot(qdti, gap, label="GDPギャップ", color="black", linestyle="dashed")
ax2.legend(loc="lower left", fontsize=16)
ax2.tick_params(labelsize=16)
ax2.set_ylabel("％", fontsize=16)



# プログラム2.6
# 単位根検定
# GDPギャップについてADF検定を実施
adf = adfuller(gap, regression="n", autolag="AIC")
print("[GDPギャップの単位根検定]")
print(f"ADF検定統計量: {adf[0]}")
print(f"p値: {adf[1]}")
print(f"臨界値: {adf[4]}")

# 実質GDPについてADF検定を実施
adf = adfuller(np.log(rqsagdp), regression="ct", autolag="AIC")
print("[四半期実質GDPの単位根検定]")
print(f"ADF検定統計量: {adf[0]}")
print(f"p値: {adf[1]}")
print(f"臨界値: {adf[4]}")

# 実質GDPの1階階差についてADF検定を実施
adf = adfuller(
    np.log(rqsagdp[1:].to_numpy()) - np.log(rqsagdp[:-1].to_numpy()),
    regression="c",
    autolag="AIC",
)
print("[四半期実質GDPの1階階差の単位根検定]")
print(f"ADF検定統計量: {adf[0]}")
print(f"p値: {adf[1]}")
print(f"臨界値: {adf[4]}")



# プログラム2.7
# BN分解
dgdp = pd.DataFrame(np.log(rqsagdp)[1:].to_numpy() - np.log(rqsagdp)[:-1].to_numpy())
ssize = len(dgdp)

L = 4  # ARモデルのラグ次数（4四半期）
y = pd.DataFrame(dgdp[(L - 1) + 1 : ssize])
x = pd.DataFrame()
for ll in range(1, L + 1):
    temp = pd.DataFrame(dgdp[L - ll : ssize + L - ll - (L - 1) - 1].to_numpy())
    x = pd.concat([x, temp], axis=1)

# ARモデルの推定
model = LinearRegression()
model.fit(x, y)
bmat = np.zeros((L, L))
bmat[0, :] = model.coef_
for i in range(1, L):
    bmat[i, i - 1] = 1
cvec, dvec = np.zeros((1, L)), np.zeros((1, L))
cvec[:, 0] = 1
dvec[:, 0] = (model.intercept_) / (1 - np.sum(model.coef_))
i_mat = np.eye(L)

delta_y = np.zeros([ssize + 1, L])
bn_trend, bn_cycle = np.zeros(ssize + 1), np.zeros(ssize + 1)
for i in range(L, ssize + 1):
    temp1 = dgdp[i - L : i].to_numpy()
    temp2 = temp1[::-1, :]
    delta_y[i] = temp2.reshape(L)
    # BN分解のトレンド成分
    bn_trend[i] = cvec @ (
        bmat @ (np.linalg.inv(i_mat - bmat)) @ (delta_y[i] - dvec).T
        + np.log(rqsagdp)[i]
    )
    # BN分解の循環成分
    bn_cycle[i] = -cvec @ bmat @ np.linalg.inv(i_mat - bmat) @ (delta_y[i] - dvec).T



#####
# グラフの描画
fig = plt.figure(figsize=(15, 13))
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
ax1.set_title("(a)実質GDPとトレンド成分", fontsize=24)
ax1.plot(qdti[L:], np.exp(bn_trend)[L:] / 1000, label="BN分解によるトレンド成分", color="black")
ax1.plot(qdti, rqsagdp / 1000, label="実質GDP", color="black", linestyle="dashed")
ax1.legend(loc="lower right", fontsize=16)
ax1.tick_params(labelsize=16)
ax1.set_ylabel("兆円", fontsize=16)
ax2.set_title("(b)実質GDPの循環成分とGDPギャップ", fontsize=24)
ax2.plot(qdti[L:], (bn_cycle[L:]) * 100, label="BN分解による循環成分", color="black")
ax2.plot(qdti, gap, label="GDPギャップ", color="black", linestyle="dashed")
ax2.legend(loc="lower left", fontsize=16)
ax2.tick_params(labelsize=16)
ax2.set_ylabel("％", fontsize=16)



# プログラム2.8
# UCモデル
mod_ucarima = sm.tsa.UnobservedComponents(np.log(rqsagdp), "rwdrift", autoregressive=2)
res_ucarima = mod_ucarima.fit(method="powell", disp=False)



#####
# グラフの描画
fig = plt.figure(figsize=(15, 13))
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
ax1.set_title("(a)実質GDPとトレンド成分", fontsize=24)
ax1.plot(qdti, np.exp(res_ucarima.level.smoothed) / 1000, label="UCモデルによるトレンド成分", color="black")
ax1.plot(qdti, rqsagdp / 1000, label="実質GDP", color="black", linestyle="dashed")
ax1.legend(loc="lower right", fontsize=16)
ax1.tick_params(labelsize=16)
ax1.set_ylabel("兆円", fontsize=16)
ax2.set_title("(b)実質GDPの循環成分とGDPギャップ", fontsize=24)
ax2.plot(qdti, (np.log(rqsagdp) - res_ucarima.level.smoothed) * 100, label="UCモデルによる循環成分", color="black")
ax2.plot(qdti, gap, label="GDPギャップ", color="black", linestyle="dashed")
ax2.legend(loc="lower left", fontsize=16)
ax2.tick_params(labelsize=16)
ax2.set_ylabel("％", fontsize=16)