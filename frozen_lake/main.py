from deepqlearning import DeepQLearning

if __name__== "__main__":
    model = DeepQLearning()
    model.train(1000, True)
    model.test(10, True)