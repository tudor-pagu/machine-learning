class ActivationFunctions:
    @staticmethod
    def sigmoid(x):
        if x >= 0:
            z = math.exp(-x)
            return 1 / (1 + z)
        else:
            z = math.exp(x) 
            return z / (1 + z)
        
    @staticmethod
    def d_sigmoid(x):
        return ActivationFunctions.sigmoid(x) * (1 - ActivationFunctions.sigmoid(x))    

    @staticmethod
    def relu(x):
        if x <= 0: 
            return 0
        else: 
            return x
    @staticmethod
    def d_relu(x):
        if x <= 0:
            return 0
        else:
            return 1
    
    @staticmethod
    def binary_cross_entropy(y,y_pred):
        return -(y*math.log(y_pred) + (1 - y) * math.log(1 - y_pred))
    
    @staticmethod
    def d_binary_cross_entropy(y,y_pred):
        return -( (y / y_pred) - ((1 - y) / (1 - y_pred)) )
    
    @staticmethod
    def mean_squared_error(y,y_pred):
        return (y_pred - y) ** 2
    
    @staticmethod
    def d_mean_squared_error(y,y_pred):
        return 2 * (y_pred - y)
    @staticmethod 
    def get_function(f):
        if f=='sigmoid':
            return ActivationFunctions.sigmoid
        if f == 'relu':
            return ActivationFunctions.relu
        if f == 'bce':
            return ActivationFunctions.binary_cross_entropy
        if f == 'mse':
            return ActivationFunctions.mean_squared_error
        
    @staticmethod
    def get_derivative(f):
        if f == 'sigmoid':
            return ActivationFunctions.d_sigmoid
        if f == 'relu':
            return ActivationFunctions.d_relu
        if f == 'bce':
            return ActivationFunctions.d_binary_cross_entropy
        if f == 'mse':
            return ActivationFunctions.d_mean_squared_error

class Layer:
    def __init__(self,num_neurons,prev_layer,activation_function):
        self.prev_layer = prev_layer
        self.activation_function = activation_function
        if prev_layer != None:
            self.w = [[0] * prev_layer.num_neurons] * num_neurons # w[cur_neuron][past_neuron]
            self.b = [0] * num_neurons
            self.dw = [[0] * prev_layer.num_neurons] * num_neurons
            self.db = [0] * num_neurons
            for i in range(len(self.w)):
                for j in range(len(self.w[i])):
                    self.w[i][j] = random.random() 
            for i in range(len(self.b)):
                self.b[i] = random.random()

        self.z = [0] * num_neurons
        self.neurons = [0] * num_neurons
        self.num_neurons = num_neurons
    def apply_gradient(self,learning_rate,nr_examples):
        if self.prev_layer == None:
            return
        
        for i in range(self.num_neurons):
            for j in range(self.prev_layer.num_neurons):
                self.w[i][j] -= learning_rate * self.dw[i][j] / nr_examples
                self.dw[i][j] = 0
        for i in range(self.num_neurons):
            self.b[i] -= learning_rate * self.db[i] / nr_examples
            self.db[i] = 0
        self.prev_layer.apply_gradient(learning_rate, nr_examples)
    def forward_propagate(self): # given a previous layer that has  
        if (self.prev_layer == None):
            return
        self.prev_layer.forward_propagate()
        for i in range(self.num_neurons):
            self.z[i] = 0
            for j in range(self.prev_layer.num_neurons):
                self.z[i] += self.w[i][j] * self.prev_layer.neurons[j]
            self.z[i] += self.b[i]
            self.neurons[i] = ActivationFunctions.get_function(self.activation_function)(self.z[i])
    def backward_propagate(self,d_a):
        if self.prev_layer == None:
            return
        for i in range(self.num_neurons):
            for j in range(self.prev_layer.num_neurons):
                self.dw[i][j] += self.prev_layer.neurons[j] * ActivationFunctions.get_derivative(self.activation_function)(self.z[i]) * d_a[i]
        for i in range(self.num_neurons):
            self.db[i] += ActivationFunctions.get_derivative(self.activation_function)(self.z[i]) * d_a[i]
        new_d_a = [0] * self.prev_layer.num_neurons
        for i in range(self.num_neurons):
            for j in range(self.prev_layer.num_neurons):
                new_d_a[j] += self.w[i][j] * ActivationFunctions.get_derivative(self.activation_function)(self.z[i]) * d_a[i]
        self.prev_layer.backward_propagate(new_d_a)

class NeuralNetwork:
    def __init__(self,loss_function):
        self.layers = []
        self.loss_function = loss_function
    
    def add_layer(self,num_neurons,activation_function):
        prev_layer = None if len(self.layers) == 0 else self.layers[-1]
        self.layers.append(Layer(num_neurons,prev_layer,activation_function))

    def forward_prop(self,x): # using the model, takes in some input data (matrix and outputs some col vector with the responses for each case)
        y = [0] * len(x)
        if len(x[0]) != self.layers[0].num_neurons:
            raise Exception("Given input that is not the same length as first layer")

        for test in range(len(x)):
            self.layers[0].neurons = copy.deepcopy(x[test])
            self.layers[-1].forward_propagate()
            y[test] = copy.deepcopy(self.layers[-1].neurons)
        return y
    
    def backward_prop(self, x, y, learning_rate):
        for test in range(len(x)):
            self.layers[0].neurons = copy.deepcopy(x[test])
            self.layers[-1].forward_propagate()
            d_a = [0] * self.layers[-1].num_neurons
            for i in range(self.layers[-1].num_neurons):
                d_a[i] = ActivationFunctions.get_derivative(self.loss_function)(y[test][i], self.layers[-1].neurons[i])
            self.layers[-1].backward_propagate(d_a)
        self.layers[-1].apply_gradient(learning_rate, len(x))
    
    def loss(self,y_pred,y):
        sum = 0
        for i in range(len(y_pred)):
            sum_per_testcase = 0
            for j in range(len(y_pred[i])):
                sum_per_testcase += ActivationFunctions.get_function(self.loss_function)(y_pred[i][j],y[i][j])
            sum_per_testcase /= len(y_pred[i])
            sum += sum_per_testcase
        sum /= len(y_pred)
        return sum      

    def train(self,x,y,num_epochs, learning_rate):
        for i in range(num_epochs):
            # --- for debugging purposes, not neccesary
            if i % 1000 == 0:
                y_pred = self.forward_prop(x)  
                l = self.loss(y,y_pred)
                print(f"epoch = {i}, l = {l}")
            # -----
            self.backward_prop(x, y, learning_rate)
