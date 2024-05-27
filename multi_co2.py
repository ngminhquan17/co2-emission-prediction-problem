import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Biến đổi dữ liệu: dựa vào lượng co2 của 5 tuần để dự đoán tuần tiếp theo
def create_ts_data(data, target, window_size = 5, target_size = 3):
    i = 1
    while i < window_size:
        data["co2_{}".format(i)] = data["co2"].shift(-i)
        i+=1

    i = 0
    while i < target_size:
        data["{}_{}".format(target_size, i)] = data["co2"].shift(-i-window_size)
        i+=1

    data = data.dropna(axis=0)
    return data

window_size = 5
target_size = 3
target = "target"

data = pd.read_csv("co2.csv")
data["time"] = pd.to_datetime(data["time"])

#Nội suy để dự đoán co2 trống
data["co2"] = data["co2"].interpolate()

#đồ thị
'''fig, ax = plt.subplots()
ax.plot(data["time"], data["co2"])
ax.set_xlabel("Year")
ax.set_ylabel("CO2")
plt.show()'''

#Biến đổi dữ liệu
data = create_ts_data(data, target, window_size, target_size)

#Phân chia dữ liệu theo chiều dọc
target_names = ["{}_{}".format(target_size, i) for i in range(target_size)]
x = data.drop(["time"] + target_names, axis=1)
y = data[target_names]

#Phân chia dữ liệu theo chiều ngang
#Không phân chia theo kiểu ngẫu nhiên mà phải lấy khoản thời gian liên tiếp
training_ratio = 0.8 #80% cho bộ train
x_train = x[:int(len(x)*training_ratio)]
y_train = y[:int(len(x)*training_ratio)]
x_test = x[int(len(x)*training_ratio):]
y_test = y[int(len(x)*training_ratio):]


regs = [LinearRegression() for _ in range(target_size)]
mae_list = []
mse_list = []
r2_list = []

for i, reg in enumerate(regs):
    reg.fit(x_train, y_train["{}_{}".format(target_size, i)])
    y_predict = reg.predict(x_test)
    mae_list.append(mean_absolute_error(y_test["{}_{}".format(target_size, i)], y_predict))
    mse_list.append(mean_squared_error(y_test["{}_{}".format(target_size, i)], y_predict))
    r2_list.append(r2_score(y_test["{}_{}".format(target_size, i)], y_predict))


'''for i, j in zip(y_predict, y_test):
    print("Predict: {}. Actual: {}".format(i, j))'''

print("MSE: {}".format(mse_list)) # càng bé càng tốt
print("MAE: {}".format(mae_list)) # càng bé càng tốt
print("R2: {}".format(r2_list)) # càng gần 1 càng tốt

