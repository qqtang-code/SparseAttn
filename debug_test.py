import time

def main():
    print("Program started. Waiting for VSCode debugger to attach...")

    # 这是一个你可以下断点的地方
    for i in range(10):
        print(f"Step {i}")
        time.sleep(1)  # 慢一点方便观察

    print("Program finished.")

if __name__ == "__main__":
    main()
