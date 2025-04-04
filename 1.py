import streamlit as st
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.optim import optimize_acqf

st.set_page_config(page_title="Carbon Black Optimization", layout="wide")
st.title("Carbon Black 조성 최적화 (단일 변수 탐색)")

# 1. 데이터 불러오기
df = pd.read_csv("slurry_data_wt%.csv")

# 2. 입력/출력 설정
x_cols = ["carbon_black_wt%"]
y_cols = ["yield_stress"]
X_raw = df[x_cols].values
Y_raw = df[y_cols].values

# 3. carbon_black_wt% 범위 설정
cb_min, cb_max = 1.75, 10.0
x_scaler = MinMaxScaler(feature_range=(0, 1))
x_scaler.fit(np.array([[cb_min], [cb_max]]))  # 명시적 범위 설정

# 4. 정규화
X_scaled = x_scaler.transform(X_raw)
train_x = torch.tensor(X_scaled, dtype=torch.double)
train_y = torch.tensor(Y_raw, dtype=torch.double)

# 5. GP 모델 학습
model = SingleTaskGP(train_x, train_y)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)

# 6. 탐색 범위 설정
input_dim = train_x.shape[1]
bounds = torch.stack([
    torch.zeros(input_dim, dtype=torch.double),
    torch.ones(input_dim, dtype=torch.double)
])

candidate_cb = None

# 7. 버튼을 눌렀을 때 추천 수행
if st.button("Candidate"):
    best_y = train_y.max().item()
    acq_fn = LogExpectedImprovement(model=model, best_f=best_y, maximize=True)

    y_pred = -float("inf")
    for attempt in range(10):
        candidate_scaled, _ = optimize_acqf(
            acq_function=acq_fn,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=100,
        )
        candidate_np = candidate_scaled.detach().numpy()
        candidate_cb = x_scaler.inverse_transform(candidate_np)[0]

        x_tensor = torch.tensor(candidate_np, dtype=torch.double)
        y_pred = model.posterior(x_tensor).mean.item()

        if y_pred > 0:
            break

    if candidate_cb is not None:
        st.subheader("추천된 Carbon Black 조성")
        st.write(f"carbon_black_wt%: **{candidate_cb[0]:.2f} wt%**")
        st.write(f"예측 Yield Stress: **{y_pred:.2f} Pa**")
    else:
        st.warning("Yield Stress > 0 조건을 만족하는 조성을 찾지 못했습니다.")

# 8. 예측 곡선 출력
x_vals_scaled = np.linspace(0, 1, 100).reshape(-1, 1)
X_test_tensor = torch.tensor(x_vals_scaled, dtype=torch.double)

model.eval()
with torch.no_grad():
    posterior = model.posterior(X_test_tensor)
    mean = posterior.mean.numpy().flatten()
    std = posterior.variance.sqrt().numpy().flatten()

cb_vals_wt = x_scaler.inverse_transform(x_vals_scaled).flatten()
train_x_wt = x_scaler.inverse_transform(train_x.numpy()).flatten()
train_y_np = train_y.numpy().flatten()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(cb_vals_wt, mean, label="Predicted Mean", color="blue")
ax.fill_between(cb_vals_wt, mean - 1.96 * std, mean + 1.96 * std, color="blue", alpha=0.2, label="95% CI")
ax.scatter(train_x_wt, train_y_np, color="red", label="Observed Data")
if candidate_cb is not None:
    pred_cb_tensor = torch.tensor(x_scaler.transform(candidate_cb.reshape(1, -1)), dtype=torch.double)
    pred_y = model.posterior(pred_cb_tensor).mean.item()
    ax.scatter(candidate_cb[0], pred_y, color="yellow", label="Candidate")

ax.set_xlabel("Carbon Black [wt%]")
ax.set_ylabel("Yield Stress [Pa]")
ax.set_title("GP Prediction (Carbon Black only)")
ax.grid(True)
ax.legend()
st.pyplot(fig)
