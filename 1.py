import streamlit as st
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
param_bounds = {
    "carbon_black_wt%": (1.75 , 10),
}

# 4. 정규화 함수
def normalize(X, bounds_array):
    return (X - bounds_array[:, 0]) / (bounds_array[:, 1] - bounds_array[:, 0])

def denormalize(X_scaled, bounds_array):
    return X_scaled * (bounds_array[:, 1] - bounds_array[:, 0]) + bounds_array[:, 0]

# 5. bounds array 생성
bounds_array = np.array([param_bounds[key] for key in x_cols])
X_scaled = normalize(X_raw, bounds_array)
train_x = torch.tensor(X_scaled, dtype=torch.double)
train_y = torch.tensor(Y_raw, dtype=torch.double)

# 6. GP 모델 학습
model = SingleTaskGP(train_x, train_y)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)

# 7. 탐색 범위 설정
input_dim = train_x.shape[1]
bounds = torch.stack([
    torch.zeros(input_dim, dtype=torch.double),
    torch.ones(input_dim, dtype=torch.double)
])

candidate_cb = None

# 8. 버튼을 눌렀을 때 추천 수행
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
        candidate_np = candidate_scaled.detach().numpy()[0]
        candidate_cb = denormalize(candidate_np.reshape(1, -1), bounds_array)[0]

        x_norm = normalize(candidate_cb.reshape(1, -1), bounds_array)
        x_tensor = torch.tensor(x_norm, dtype=torch.double)
        y_pred = model.posterior(x_tensor).mean.item()

        if y_pred > 0:
            break

    if candidate_cb is not None:
        st.subheader("추천된 Carbon Black 조성")
        st.write(f"carbon_black_wt%: **{candidate_cb[0]:.2f} wt%**")
        st.write(f"예측 Yield Stress: **{y_pred:.2f} Pa**")
    else:
        st.warning("Yield Stress > 0 조건을 만족하는 조성을 찾지 못했습니다.")

# 9. 예측 곡선 출력
dx = 0
x_vals = np.linspace(0, 1, 100)
mean_scaled = np.mean(X_scaled, axis=0)
X_test_scaled = np.tile(mean_scaled, (100, 1))
X_test_scaled[:, dx] = x_vals
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.double)

model.eval()
with torch.no_grad():
    posterior = model.posterior(X_test_tensor)
    mean = posterior.mean.numpy().flatten()
    std = posterior.variance.sqrt().numpy().flatten()

cb_vals_wt = denormalize(X_test_scaled, bounds_array)[:, dx]
train_x_wt = denormalize(train_x.numpy(), bounds_array)[:, dx]
train_y_np = train_y.numpy().flatten()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(cb_vals_wt, mean, label="Predicted Mean", color="blue")
ax.fill_between(cb_vals_wt, mean - 1.96 * std, mean + 1.96 * std, color="blue", alpha=0.2, label="95% CI")
ax.scatter(train_x_wt, train_y_np, color="red", label="Observed Data")
if candidate_cb is not None:
    ax.scatter(
        candidate_cb[0],
        model.posterior(torch.tensor([normalize(candidate_cb.reshape(1, -1), bounds_array)], dtype=torch.double)).mean.item(),
        color="yellow", label="Candidate"
    )
ax.set_xlabel("Carbon Black [wt%]")
ax.set_ylabel("Yield Stress [Pa]")
ax.set_title("GP Prediction (Carbon Black only)")
ax.grid(True)
ax.legend()
st.pyplot(fig)