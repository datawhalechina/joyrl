[multiprocessing官方教程-Python](https://docs.python.org/zh-cn/3/library/multiprocessing.html)

容易陷入的误区：

* 电脑的CPU核数不等于支持的进程数，实际上能够支持的进程数更多，一般每个核支持两个进程
* 进程与线程也有区别

执行下列代码可查看电脑能够支持的最大进程数：
```python
import multiprocessing as mp
print(mp.cpu_count())
```

## 构建子进程的方式

一般有三种，即fork，spawn和forkserver。unix环境中默认为fork，win环境下不支持fork，需要设置为spawn。

fork模式下，除了必要的启动资源，子进程中的其他变量、包和数据等等都继承父进程，因而启动较快，但是大部分用的都是父进程的数据，不是很安全的模式

spawn模式下，子进程是从头开始创建的，变量、包和数据等等都是从父进程拷贝而来，因此启动较慢，但是安全系数高。

```python
import multiprocessing as mp
print(mp.get_all_start_methods()) # 查看所有启动子进程的方法
print(mp.get_start_method()) # 查看当前系统启动子进程的默认方法
```

