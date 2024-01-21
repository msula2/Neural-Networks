from math import exp
class Perceptron(object):

	# Create a new Perceptron
	# 
	# Params:	bias -	arbitrarily chosen value that affects the overall output
	#					regardless of the inputs
	#
	#			synaptic_weights -	list of initial synaptic weights for this Perceptron
	def __init__(self, bias, synaptic_weights):
		
		self.bias = bias
		self.synaptic_weights = synaptic_weights


	# Activation function
	#	Quantizes the induced local field
	#
	# Params:	z - the value of the indiced local field
	#
	# Returns:	an integer that corresponds to one of the two possible output values (usually 0 or 1)
	def activation_function(self, z):
		
		#Sigmoid activation function
		threshold = 0.5
		sig_prob = 1.0 / (1.0 + exp(-z))
		if (sig_prob > threshold):
			return 1
		else:
			return 0


	# Compute and return the weighted sum of all inputs (not including bias)
	#
	# Params:	inputs - a single input vector (which may contain multiple individual inputs)
	#
	# Returns:	a float value equal to the sum of each input multiplied by its
	#			corresponding synaptic weight
	def weighted_sum_inputs(self, inputs):
		weighted_sum = 0
		for i in range(0, len(inputs) - 1):
			weighted_sum += inputs[i] * self.synaptic_weights[i]
		return weighted_sum
		

	# Compute the induced local field (the weighted sum of the inputs + the bias)
	#
	# Params:	inputs - a single input vector (which may contain multiple individual inputs)
	#
	# Returns:	the sum of the weighted inputs adjusted by the bias
	def induced_local_field(self, inputs):
		weighted_sum = self.weighted_sum_inputs(inputs)
		z = weighted_sum + self.bias
		return z

	# Predict the output for the specified input vector
	#
	# Params:	input_vector - a vector or row containing a collection of individual inputs
	#
	# Returns:	an integer value representing the final output, which must be one of the two
	#			possible output values (usually 0 or 1)
	def predict(self, input_vector):
		z = self.induced_local_field(input_vector)
		hard_limiter = self.activation_function(z)
		return hard_limiter


	# Train this Perceptron
	#
	# Params:	training_set - a collection of input vectors that represents a subset of the entire dataset
	#			learning_rate_parameter - 	the amount by which to adjust the synaptic weights following an
	#										incorrect prediction
	#			number_of_epochs -	the number of times the entire training set is processed by the perceptron
	#
	# Returns:	no return value
	def train(self, training_set, learning_rate_parameter, number_of_epochs):
		print("------------Training------------")
		for epoch in range(0, number_of_epochs):
			print("For Epoch: ", epoch)
			correct_outputs = 0
			for i in range(0, len(training_set)):
				prediction = self.predict(training_set[i])
				actual_output = training_set[i][-1]
				error = actual_output - prediction
				#If error is zero, no updates occur
				for j in range(len(self.synaptic_weights)):
					self.synaptic_weights[j] = self.synaptic_weights[j] + (learning_rate_parameter * error * training_set[i][j])
				#Correct prediction, do not update weights
				if(error == 0):
					correct_outputs += 1
			accuracy = (correct_outputs / len(training_set)) * 100
			print("Accuracy: " ,round(accuracy,2), "%")  
	# Test this Perceptron
	# Params:	test_set - the set of input vectors to be used to test the perceptron after it has been trained
	#
	# Returns:	a collection or list containing the actual output (i.e., prediction) for each input vector
	def test(self, test_set):
		print("------------Testing------------")
		prediction = list()
		for i in range(0, len(test_set)):
				prediction.append(self.predict(test_set[i]))
		return prediction

