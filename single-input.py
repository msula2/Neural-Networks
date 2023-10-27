input = 0.2
goal_pred = 6
weight = 0

print("------------------Intializing Neural Network------------------")
print("Goal: ", goal_pred)
print("Input: ", input)
print("--------------------------------------------------------------")
for iteration in range(200):
    print("Iteration #%d" %iteration)
    pred = input * weight
    print("* Prediction: ",pred)
    error = (pred - goal_pred) ** 2
    print("* Error: ",error)
    delta = pred - goal_pred
    weight_delta = input * delta
    weight = weight - weight_delta