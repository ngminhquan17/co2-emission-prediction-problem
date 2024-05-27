import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Biến đổi dữ liệu: dựa vào lượng co2 của 5 tuần để dự đoán tuần tiếp theo
def create_ts_data(data, target, window_size = 5):
    i = 1
    while i < window_size:
        data["co2_{}".format(i)] = data["co2"].shift(-i)
        i+=1
    data[target] = data["co2"].shift(-i)
    data = data.dropna(axis=0)
    return data

window_size = 5
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
data = create_ts_data(data,target, window_size)

#Phân chia dữ liệu theo chiều dọc
x = data.drop([target, "time"], axis=1)
y = data[target]

#Phân chia dữ liệu theo chiều ngang
#Không phân chia theo kiểu ngẫu nhiên mà phải lấy khoản thời gian liên tiếp
training_ratio = 0.8 #80% cho bộ train
x_train = x[:int(len(x)*training_ratio)]
y_train = y[:int(len(x)*training_ratio)]
x_test = x[int(len(x)*training_ratio):]
y_test = y[int(len(x)*training_ratio):]

reg = LinearRegression()
reg.fit(x_train, y_train)
y_predict = reg.predict(x_test)


'''for i, j in zip(y_predict, y_test):
    print("Predict: {}. Actual: {}".format(i, j))'''

print("MSE: {}".format(mean_squared_error(y_test, y_predict))) # càng bé càng tốt
print("MAE: {}".format(mean_absolute_error(y_test, y_predict))) # càng bé càng tốt
print("R2: {}".format(r2_score(y_test, y_predict))) # càng gần 1 càng tốt

