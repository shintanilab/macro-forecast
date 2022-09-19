# プログラム1.1
import math

import numpy as np
import pandas as pd
import scipy as sp
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from matplotlib.pylab import rcParams
rcParams["figure.figsize"] = 15, 6
import japanize_matplotlib



# プログラム1.2
# データの読み込み
gap = pd.read_excel("data_1.xlsx", sheet_name="GDPギャップ(内閣府)", header=[0])
dlen = len(gap)
caogap = gap.iloc[0:dlen, 1]
dti = pd.date_range("1980-01-01", periods=dlen, freq="QS")
# データの出力
print(gap)

# グラフの描画
plt.plot(dti, caogap, color="black")
plt.hlines([0], dti[0], dti[-1], color="black", linestyles="dashed")
plt.tick_params(labelsize=16)
plt.xlim(dti[0], dti[-1])
plt.ylabel("％", fontsize=16)



# プログラム1.3
# AR(1)モデルの推定
model = sm.tsa.AutoReg(caogap, lags=1)
result = model.fit()
print(result.summary())



# プログラム1.4
# コレログラムの計算
H = 30  # ラグ次数（30四半期）
CONFIDENCE_LEVEL = 0.95  # 有意水準
autoc, se = np.zeros(H), np.zeros(H)
lbound, ubound = sp.stats.norm.interval(alpha=CONFIDENCE_LEVEL, loc=0, scale=1)
for i in range(1, H + 1):
    x, y = pd.DataFrame(caogap[i:].to_numpy()), pd.DataFrame(caogap[:-i].to_numpy())
    model = LinearRegression()
    result = model.fit(x, y)
    autoc[i - 1] = result.coef_
    se[i - 1] = (
        1 / len(caogap) * (1 + 2 * sum([autoc[x] ** 2 for x in range(i)]))
    ) ** (1 / 2)

# グラフの描画
hlabel = pd.DataFrame(range(1, H + 1))
plt.plot(np.ravel(hlabel), autoc, color="black", label="自己相関係数", marker="o")
plt.hlines([0], 0, H + 1, color="black", linestyles="dashed")
plt.fill_between(
    np.ravel(hlabel), ubound * se, lbound * se, color="grey", alpha=0.5, label="±1.96×SE区間"
)
plt.legend(loc="upper right", fontsize=16)
plt.tick_params(labelsize=16)
plt.xlim(0, H + 1)
plt.xlabel("ラグ次数", fontsize=16)



# プログラム1.5
# Q検定、修正Q検定
LAG = 8  # ラグ次数（8四半期）
rho_lag, q_bp, q_lb, crit = np.zeros(LAG), np.zeros(LAG), np.zeros(LAG), np.zeros(LAG)
for i in range(1, LAG + 1):
    data_x, data_y = pd.DataFrame(caogap[: dlen - i]), pd.DataFrame(caogap[i:])
    model_lag = LinearRegression(fit_intercept=True)
    model_lag.fit(data_x, data_y)
    rho_lag[i - 1] = model_lag.coef_
    # Q検定統計量
    q_bp[i - 1] = dlen * sum(rho_lag**2)
    # 修正Q検定統計量
    q_lb[i - 1] = dlen * (dlen + 2) * sum(rho_lag**2 / (dlen - i))
    # カイ2乗分布の臨界値
    crit[i - 1] = sp.stats.chi2.ppf(q=0.95, df=i)
# 結果表の作成
df_table = pd.DataFrame(
    [rho_lag, q_bp, q_lb, crit],
    columns=[f"lag={i+1}" for i in range(LAG)],
    index=["相関係数", "Q_BP", "Q_LB", "臨界値"],
)
print(df_table)



# プログラム1.6
# AR(1)モデルによる予測と予測誤差（反復予測）
L = 1  # ARモデルのラグ次数（1四半期）
MAX_H = 9  # 最大予測期間（9四半期）
CONFIDENCE_LEVEL = 0.95  # 区間予測の信頼係数

# 1980年～2018年までのデータを用いてARモデルを推定
dti = pd.date_range("1980-01-01", periods=dlen, freq="QS")
ssize = sum(dti.year < 2019)
lbound, ubound = sp.stats.norm.interval(alpha=CONFIDENCE_LEVEL, loc=0, scale=1)
predict_result, conf_result = np.zeros(MAX_H), np.zeros(MAX_H)
for h in range(MAX_H):
    for l in range(1, L + 1):
        train_x, train_y = pd.DataFrame(), pd.DataFrame(caogap[L : ssize])
        test_x = pd.DataFrame()
        for ll in range(1, l + 1):
            train_x_lagged = pd.DataFrame(
                caogap[L - ll : L - ll + len(train_y)].to_numpy()
            )
            train_x = pd.concat([train_x, train_x_lagged], axis=1)
        for ll in range(1, l + 1):
            test_x_lagged = pd.DataFrame(
                caogap[
                    ssize + l - ll - (l - 1) - 1 : ssize + l - ll - (l - 1)
                ].to_numpy()
            )
            test_x = pd.concat([test_x, test_x_lagged], axis=1)
    # ARモデルの推定
    model = LinearRegression()
    model.fit(train_x, train_y)
    # 予測値の計算
    for _ in range(h + 1):
        y_hat = model.predict(train_x)
        forecast = pd.DataFrame(
            np.sum((test_x * model.coef_).to_numpy()) + model.intercept_
        )
        test_x = pd.concat(
            [forecast, pd.DataFrame((test_x.T[:-1]).T.to_numpy())], axis=1
        )
    predict_result[h] = forecast.to_numpy()
    # 予測の信頼区間の計算
    se = np.std(train_y.to_numpy() - y_hat)
    bmat = np.zeros((L, L))
    bmat[0, :] = model.coef_
    for i in range(1, L):
        bmat[i, i - 1] = 1
    smat = np.zeros((L, L))
    smat[0, 0] = 1
    cvec = np.zeros((1, L))
    cvec[:, 0] = 1
    coef = sum(
        cvec
        @ np.linalg.matrix_power(bmat, x)
        @ smat
        @ np.linalg.matrix_power(bmat, x)
        @ cvec.T
        for x in range(h + 1)
    )
    conf_result[h] = coef ** (1 / 2) * se * ubound
ar_forecast = (
    pd.concat([caogap[0:ssize], pd.DataFrame(predict_result)], axis=0)
).to_numpy()
ar_bound = (
    pd.concat([pd.DataFrame(np.zeros(ssize)), pd.DataFrame(conf_result)], axis=0)
).to_numpy()
ar_ubound, ar_lbound = ar_forecast + ar_bound, ar_forecast - ar_bound

# グラフの描画
plt.title("GDPギャップ", fontsize=24)
plt.plot(dti, caogap, label="実績値", color="black")
plt.plot(
    dti[ssize - 1 : ssize + MAX_H],
    ar_forecast[ssize - 1 : ssize + MAX_H, :],
    label="AR(1)予測〈反復予測〉",
    color="black",
    linestyle="dashed",
)
plt.fill_between(
    np.ravel(dti[ssize - 1 : ssize + MAX_H]),
    np.ravel(ar_ubound[ssize - 1 : ssize + MAX_H, :]),
    np.ravel(ar_lbound[ssize - 1 : ssize + MAX_H, :]),
    color="grey",
    alpha=0.5,
    label="95％信頼区間",
)
plt.legend(loc="lower left", fontsize=16)
plt.tick_params(labelsize=16)
plt.ylabel("％", fontsize=16)



# プログラム1.7
# AR(1)モデルによる予測と予測誤差（直接予測）
L = 1  # ARモデルのラグ次数（1四半期）
MAX_H = 9  # 最大予測期間（9四半期）
CONFIDENCE_LEVEL = 0.95  # 区間予測の信頼係数

# 1980年～2018年までのデータを用いてARモデルを推定
dti = pd.date_range("1980-01-01", periods=dlen, freq="QS")
ssize = sum(dti.year < 2019)
lbound, ubound = sp.stats.norm.interval(alpha=CONFIDENCE_LEVEL, loc=0, scale=1)
predict_result, conf_result = np.zeros(MAX_H), np.zeros(MAX_H)
for h in range(MAX_H):
    for l in range(1, L + 1):
        train_x, train_y = pd.DataFrame(), pd.DataFrame(caogap[L + h : ssize])
        test_x = pd.DataFrame()
        for ll in range(1, l + 1):
            train_x_lagged = pd.DataFrame(
                caogap[L - ll : L - ll + len(train_y)].to_numpy()
            )
            train_x = pd.concat([train_x, train_x_lagged], axis=1)
        for ll in range(1, l + 1):
            test_x_lagged = pd.DataFrame(
                caogap[
                    ssize + l - ll - (l - 1) - h - 1 : ssize + l - ll - (l - 1) - h
                ].to_numpy()
            )
            test_x = pd.concat([test_x, test_x_lagged], axis=1)
    # ARモデルの推定
    model = LinearRegression()
    model.fit(train_x, train_y)
    # 予測値の計算
    predict_result[h] = model.predict(test_x)
    # 予測の信頼区間の計算
    se = np.std(train_y.to_numpy() - model.predict(train_x))
    conf_result[h] = se * ubound
ar_forecast = pd.concat([caogap[0:ssize], pd.DataFrame(predict_result)], axis=0).to_numpy()
ar_bound = pd.concat(
    [pd.DataFrame(np.zeros(ssize)), pd.DataFrame(conf_result)], axis=0
).to_numpy()
ar_ubound, ar_lbound = ar_forecast + ar_bound, ar_forecast - ar_bound



#####
# グラフの描画
plt.title("GDP ギャップ", fontsize=24)
plt.plot(dti, caogap, label="実績値", color="black")
plt.plot(
    dti[ssize - 1 : ssize + MAX_H],
    ar_forecast[ssize - 1 : ssize + MAX_H, :],
    label="AR(1)予測〈直接予測〉",
    color="black",
    linestyle="dashed",
)
plt.fill_between(
    np.ravel(dti[ssize - 1 : ssize + MAX_H]),
    np.ravel(ar_ubound[ssize - 1 : ssize + MAX_H, :]),
    np.ravel(ar_lbound[ssize - 1 : ssize + MAX_H, :]),
    color="grey",
    alpha=0.5,
    label="95％信頼区間",
)
plt.legend(loc="lower left", fontsize=16)
plt.tick_params(labelsize=16)
plt.ylabel("％", fontsize=16)



# プログラム1.8
# GDPギャップのARモデルのラグ次数の選択
def get_aic_bic(caogap, y, y_hat, param1, param2):
    sse = ((y - y_hat) ** 2).sum(axis=0)
    aic = math.log(sse / (len(caogap) - param1)) + (l + 1) * 2 / (len(caogap) - param2)
    bic = math.log(sse / (len(caogap) - param1)) + (l + 1) * math.log(
        (len(caogap) - param2)
    ) / (len(caogap) - param2)
    return aic, bic


MAX_L = 4  # ARモデルの最大ラグ次数（4四半期）
aic_result, bic_result = np.zeros([MAX_L + 1, 3]), np.zeros([MAX_L + 1, 3])
# AR(0)モデルの場合
l = 0
# AIC・BICの計算方法1:推定するすべてのARモデルのサンプルサイズが同じになるよう調整した場合
y = pd.DataFrame(caogap[MAX_L:])
aic_result[l, 0], bic_result[l, 0] = get_aic_bic(caogap, y, np.mean(y.to_numpy()), MAX_L, MAX_L)
# AIC・BICの計算方法2:推定するすべてのARモデルのサンプルサイズが同じになるよう調整した場合（標本分散の自由度を調整）
y = pd.DataFrame(caogap[MAX_L:])
aic_result[l, 1], bic_result[l, 1] = get_aic_bic(
    caogap, y, np.mean(y.to_numpy()), MAX_L + l + 1, MAX_L
)
# AIC・BICの計算方法3:それぞれのARモデルで利用できる観測値をすべて利用した場合
y = pd.DataFrame(caogap[l:])
aic_result[l, 2], bic_result[l, 2] = get_aic_bic(caogap, y, np.mean(y.to_numpy()), l, l)

# AR(p)モデル（p>0）の場合
for l in range(1, MAX_L + 1):
    # AIC・BICの計算方法1:推定するすべてのARモデルのサンプルサイズが同じになるよう調整した場合
    x, y = pd.DataFrame(), pd.DataFrame(caogap[MAX_L:])
    for ll in range(1, l + 1):
        x_lagged = pd.DataFrame(caogap[MAX_L - ll : MAX_L - ll + len(y)].to_numpy())
        x = pd.concat([x, x_lagged], axis=1)
    model = LinearRegression()
    model.fit(x, y)
    aic_result[l, 0], bic_result[l, 0] = get_aic_bic(caogap, y, model.predict(x), MAX_L, MAX_L)

    # AIC・BICの計算方法2:推定するすべてのARモデルのサンプルサイズが同じになるよう調整した場合（誤差項の標本分散の自由度を調整）
    x, y = pd.DataFrame(), pd.DataFrame(caogap[MAX_L:])
    for ll in range(1, l + 1):
        x_lagged = pd.DataFrame(caogap[MAX_L - ll : MAX_L - ll + len(y)].to_numpy())
        x = pd.concat([x, x_lagged], axis=1)
    model = LinearRegression()
    model.fit(x, y)
    aic_result[l, 1], bic_result[l, 1] = get_aic_bic(
        caogap, y, model.predict(x), MAX_L + l + 1, MAX_L
    )

    # AIC・BICの計算方法3:それぞれのARモデルで利用できる観測値をすべて利用した場合
    x, y = pd.DataFrame(), pd.DataFrame(caogap[l:])
    for ll in range(1, l + 1):
        x_lagged = pd.DataFrame(caogap[l - ll : l - ll + len(y)].to_numpy())
        x = pd.concat([x, x_lagged], axis=1)
    model = LinearRegression()
    model.fit(x, y)
    aic_result[l, 2], bic_result[l, 2] = get_aic_bic(caogap, y, model.predict(x), l, l)

# 結果表の作成
df_aic_result = pd.DataFrame(
    aic_result,
    columns=[f"方法{i+1}" for i in range(3)],
    index=[f"L={i}" for i in range(MAX_L + 1)],
)
df_bic_result = pd.DataFrame(
    bic_result,
    columns=[f"方法{i+1}" for i in range(3)],
    index=[f"L={i}" for i in range(MAX_L + 1)],
)
print("[AIC]\n", df_aic_result)
print("[BIC]\n", df_bic_result)
print("[best AIC lag length]\n", df_aic_result.idxmin())
print("[best BIC lag length]\n", df_bic_result.idxmin())



# プログラム1.9
# AR(1)モデルの誤差項のQ検定、修正Q検定
L = 1  # ARモデルのラグ次数
M = 8  # 系列相関の次数
model = sm.tsa.AutoReg(caogap, lags=L)
result = model.fit()
pred = result.predict()
resid = caogap[L:] - pred
dlen = len(resid)
rho_lag = np.zeros(M)
q_bp, q_lb, crit_temp, crit = np.zeros(M), np.zeros(M), np.zeros(M), np.zeros(M)
for i in range(1, M + 1):
    data_x, data_y = pd.DataFrame(resid[L : dlen - i]), pd.DataFrame(resid[L + i :])
    model_lag = LinearRegression(fit_intercept=True)
    model_lag.fit(data_x, data_y)
    rho_lag[i - 1] = model_lag.coef_
    # Q検定統計量
    q_bp[i - 1] = dlen * sum(rho_lag**2)
    # 修正Q検定統計量
    q_lb[i - 1] = dlen * (dlen + 2) * sum(rho_lag**2 / (dlen - i))
    # カイ二乗分布の臨界値の計算
    crit_temp[i - 1] = sp.stats.chi2.ppf(q=0.95, df=i)
crit[L:] = crit_temp[:M-L]
crit[:L] ="nan"

# 結果表の作成
df_table = pd.DataFrame(
    [rho_lag, q_bp, q_lb, crit],
    columns=[f"m={i+1}" for i in range(M)],
    index=["相関係数", "Q_BP", "Q_LB", "臨界値"],
)
print(df_table)



# プログラム1.10
# AR(p)モデルによる予測と予測誤差
MAX_L = 4  # ARモデルの最大ラグ次数（4四半期）
MAX_H = 9  # 最大予測期間（9四半期）
CONFIDENCE_LEVEL = 0.95  # 区間予測の信頼係数

# 1980年～2018年までのデータを用いてARモデルを推定
ssize = sum(dti.year < 2019)
lbound, ubound = sp.stats.norm.interval(alpha=CONFIDENCE_LEVEL, loc=0, scale=1)
# サンプルサイズ
nsample = len(caogap)
# AIC、BICにより選択されたラグ次数
best_aic_lag, best_bic_lag = np.zeros(MAX_H), np.zeros(MAX_H)
best_aic_predict, best_bic_predict = np.zeros(MAX_H), np.zeros(MAX_H)
best_aic_conf, best_bic_conf = np.zeros(MAX_H), np.zeros(MAX_H)
for h in range(MAX_H):
    predict_result, conf_result, aic_result, bic_result = (
        np.zeros(MAX_L),
        np.zeros(MAX_L),
        np.zeros(MAX_L),
        np.zeros(MAX_L),
    )
    for l in range(1, MAX_L + 1):
        train_x, train_y = pd.DataFrame(), pd.DataFrame(caogap[MAX_L + h : ssize])
        test_x = pd.DataFrame()
        for ll in range(1, l + 1):
            train_x_lagged = pd.DataFrame(
                caogap[MAX_L - ll : MAX_L - ll + len(train_y)].to_numpy()
            )
            train_x = pd.concat([train_x, train_x_lagged], axis=1)
        for ll in range(1, l + 1):
            test_x_lagged = pd.DataFrame(
                caogap[
                    ssize + l - ll - (l - 1) - h - 1 : ssize + l - ll - (l - 1) - h
                ].to_numpy()
            )
            test_x = pd.concat([test_x, test_x_lagged], axis=1)
        # ARモデルの推定
        model = LinearRegression()
        model.fit(train_x, train_y)
        y_hat = model.predict(train_x)
        se = np.std(train_y.to_numpy() - y_hat)
        sse = ((train_y[MAX_L - l :] - y_hat[MAX_L - l :]) ** 2).sum(axis=0)
        aic = math.log(sse / len(train_y)) + (l + 1) * 2 / len(train_y)
        bic = math.log(sse / len(train_y)) + (l + 1) * math.log(len(train_y)) / len(
            train_y
        )
        forecast = model.predict(test_x)
        conf_result[l - 1], aic_result[l - 1], bic_result[l - 1], predict_result[l - 1] = (
            se * ubound,
            aic,
            bic,
            forecast,
        )
    best_aic_lag[h], best_bic_lag[h] = (pd.DataFrame(aic_result)).idxmin() + 1, (
        pd.DataFrame(bic_result)
    ).idxmin() + 1
    best_aic_predict[h], best_bic_predict[h] = (
        predict_result[(pd.DataFrame(aic_result)).idxmin()],
        predict_result[(pd.DataFrame(bic_result)).idxmin()],
    )
    best_aic_conf[h], best_bic_conf[h] = (
        conf_result[(pd.DataFrame(aic_result)).idxmin()],
        conf_result[(pd.DataFrame(bic_result)).idxmin()],
    )

ar_forecast1 = (
    pd.concat([caogap[0:ssize], pd.DataFrame(best_aic_predict)], axis=0)
).to_numpy()
ar_bound1 = (
    pd.concat([pd.DataFrame(np.zeros(ssize)), pd.DataFrame(best_aic_conf)], axis=0)
).to_numpy()
ar_ubound1, ar_lbound1 = ar_forecast1 + ar_bound1, ar_forecast1 - ar_bound1

ar_forecast2 = (
    pd.concat([caogap[0:ssize], pd.DataFrame(best_bic_predict)], axis=0)
).to_numpy()
ar_bound2 = (
    pd.concat([pd.DataFrame(np.zeros(ssize)), pd.DataFrame(best_bic_conf)], axis=0)
).to_numpy()
ar_ubound2, ar_lbound2 = ar_forecast2 + ar_bound2, ar_forecast2 - ar_bound2



#####
# グラフの描画
fig = plt.figure(figsize=(15, 20))
ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3)
ax1.set_title("(a)GDPギャップ(AICによるラグ次数選択)", fontsize=24)
ax1.plot(dti, caogap, label="実績値", color="black")
ax1.plot(
    dti[ssize - 1 : ssize + MAX_H],
    ar_forecast1[ssize - 1 : ssize + MAX_H, :],
    label="AR(p)予測〈直接予測〉",
    color="black",
    linestyle="dashed",
)
ax1.fill_between(
    np.ravel(dti[ssize - 1 : ssize + MAX_H]),
    np.ravel(ar_ubound1[ssize - 1 : ssize + MAX_H, :]),
    np.ravel(ar_lbound1[ssize - 1 : ssize + MAX_H, :]),
    color="grey",
    alpha=0.5,
    label="95％信頼区間",
)
ax1.legend(loc="lower left", fontsize=16)
ax1.tick_params(labelsize=16)
ax1.set_ylabel("％", fontsize=16)
ax2.set_title("(b)GDPギャップ(BICによるラグ次数選択)", fontsize=24)
ax2.plot(dti, caogap, label="実績値", color="black")
ax2.plot(
    dti[ssize - 1 : ssize + MAX_H],
    ar_forecast2[ssize - 1 : ssize + MAX_H, :],
    label="AR(p)予測〈直接予測〉",
    color="black",
    linestyle="dashed",
)
ax2.fill_between(
    np.ravel(dti[ssize - 1 : ssize + MAX_H]),
    np.ravel(ar_ubound2[ssize - 1 : ssize + MAX_H, :]),
    np.ravel(ar_lbound2[ssize - 1 : ssize + MAX_H, :]),
    color="grey",
    alpha=0.5,
    label="95％信頼区間",
)
ax2.legend(loc="lower left", fontsize=16)
ax2.tick_params(labelsize=16)
ax2.set_ylabel("％", fontsize=16)
ax3.set_title("(c)選択されたラグ次数", fontsize=24)
ax3.plot(
    pd.DataFrame(x for x in range(1, MAX_H + 1)).to_numpy(),
    pd.DataFrame(best_aic_lag).to_numpy(),
    label="AIC",
    color="black")
ax3.plot(
    pd.DataFrame(x for x in range(1, MAX_H + 1)).to_numpy(),
    pd.DataFrame(best_bic_lag).to_numpy(),
    label="BIC",
    color="black",
    linestyle="dashed",
)
ax3.legend(loc="upper right", fontsize=16)
ax3.tick_params(labelsize=16)
ax3.set_xlabel("予測期間", fontsize=16)
ax3.set_ylabel("ラグ次数", fontsize=16)
ax3.set_ylim(0, MAX_L + 1)