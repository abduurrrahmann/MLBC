from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
X=[[1],[2],[3],[4],[5]]
y=[2,4,6,8,10]
model=LinearRegression()
model.fit(X,y)
y_pred=model.predict(X)
plt.scatter(X,y,color='blue',label='Actual Data')
plt.plot(X,y_pred,color='red',label='Best Fit Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()