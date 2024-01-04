from memory_profiler import profile

class MyClass:
    @profile
    def my_method(self, n):
        # 示例：创建一个大列表来模拟内存使用
        a = [i for i in range(n)]
        return a

def main():
    print('start')
    instance = MyClass()
    # method_name = 'my_method'
    n = 1000000  # 示例参数

    # 使用 getattr 来动态调用方法
    result = instance.my_method(n)
    # result = getattr(instance, method_name)(n)
    print(result[:10])  # 打印部分结果以验证

if __name__ == "__main__":
    main()
