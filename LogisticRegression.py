def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)
    
class LinearRegression:
    def __init__(self,n_features,regularization=0):
        self.n_features = n_features
        self.w = np.zeros((n_features,1))
        self.b = 0
        self.regularization = regularization

    def forward(self,x):
        return np.vectorize(sigmoid)(np.matmul(x,self.w) + self.b)
    def loss(self,y,y_pred):
        sum = 0
        for i in range(0,len(y)):
            sum += y[i][0] * np.log(y_pred[i][0]) + (1 - y[i][0]) * np.log(1 - y_pred[i][0])
        sum /= len(y)
        sum *= -1
        
        if self.regularization != 0:
            reg = 0
            for weight in self.w:
                reg += (weight[0] ** 2)
            reg = reg * self.regularization / (2 * self.n_features)
            sum += reg
        return sum
    
    def gradient_w(self,x,y,y_pred):
        return np.transpose(np.matmul(np.transpose(y_pred - y),x)/(len(x))) + (self.regularization / self.n_features ) * self.w
    def gradient_b(self, y,y_pred):
        return (y_pred - y).mean()
    def train(self,x,y,nr_epochs,learning_rate):
        for epoch in range(nr_epochs):
            y_pred = self.forward(x)
            dw = self.gradient_w(x,y,y_pred)
            db = self.gradient_b(y,y_pred)
            self.w = self.w - learning_rate * dw
            self.b = self.b - learning_rate * db
    def get_loss(self,x,y):
        y_pred = self.forward(x)
        l = self.loss(y,y_pred)
        return l
    def get_accuracy(self,x,y):
        y_pred = self.forward(x)
        nr_correct = 0
        for i in range(len(y_pred)):
            ans = 0 if y_pred[i][0] < 0.5 else 1
            if (ans == y[i][0]):
                nr_correct += 1
        return nr_correct/len(x)

