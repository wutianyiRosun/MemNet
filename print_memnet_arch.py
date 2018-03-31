from memnet1 import MemNet
def main():
    print("===> Building model")
    model = MemNet(1, 64, 6, 6)
    print(model)
if __name__ == "__main__":
    main()
