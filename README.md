# 数据科学(Data Science)

> 本仓库的代码全部在 Anaconda 的 jupyter NoteBook 中编写

> 数据科学工作流
> 1. Inquire：通过数据获取什么信息，解决什么问题
> 2. Obtain：数据如何获取（[Python](https://www.python.org/)）
> 3. Scrub：数据的清理、整理和分析（[pandas](https://pandas.pydata.org)、[NumPy](http://www.numpy.org)）
> 4. Explore：展示数据、数据可视化（[MatPlotLib](https://matplotlib.org)、[Seaborn](http://seaborn.pydata.org/#)）
> 5. Model：通过数据分析和机器学习对数据进行进一步分析和训练（[SciKit-learn](http://scikit-learn.org/)、[SciPy](https://www.scipy.org)、[TensorFlow](https://tensorflow.google.cn)）
> 6. iNterpret：最终结果展示（[BOKEH](https://bokeh.pydata.org/en/latest/)、[D3.js](https://d3js.org)）

# 目录（Contents）

* [1. 环境搭建](#1-环境搭建)
* [2. NumPy 入门](#2-numpy-入门)
    * [2.1 NumPy 中创建数组及访问数组的元素](#21-numpy-中创建数组及访问数组的元素)
        * [2.1.1 创建数组](#211-创建数组)
        * [2.1.2 访问数组元素](#212-访问数组元素)
        * [2.1.3 创建矩阵](#213-创建矩阵)
    * [2.2 数组和矩阵运算](#22-数组和矩阵运算)
        * [2.2.1 数组的运算](#221-数组的运算)
        * [2.2.2 矩阵的运算](#222-矩阵的运算)
    * [2.3 Array 的 input 和 output](#23-array-的-input-和-output)
        * [2.3.1 使用 pickle 序列化 numpy array](#231-使用-pickle-序列化-numpy-array)
        * [2.3.2 使用 numpy 序列化 numpy array](#232-使用-numpy-序列化-numpy-array)
* [3. pandas 入门](#3-pandas-入门)
* [4. pandas 玩转数据](#4-pandas-玩转数据)
* [5. 绘图与可视化-MatPlotLib](#5-绘图与可视化-matplotlib)
* [6. 绘图与可视化-SeaBorn](#6-绘图与可视化-seaborn)
* [7. 数据分析项目实战](#7-数据分析项目实战)

---

* [附录](#附录)
    * [1. 数据可续领域5个组建 Python 库](#1-数据科学领域5个组建-python-库)
    * [2. 数学基础-矩阵运算](#2-数学基础-矩阵运算)

---

## 1. 环境搭建

各种平台使用 [Anaconda](https://www.anaconda.com/download/#macos) 搭建数据分析开发环境

环境搭建可参考 	
Timothy老师的 [搭建开发环境，跨平台原理](http://sa.mentorx.net/course/54/task/321/show)

## 2. NumPy 入门

高效的数据结构 Array（数组）介绍，是学习其他高级数据结构的基础

[NumPy Docs](https://docs.scipy.org/doc/numpy/user/quickstart.html)

### 2.1 NumPy 中创建数组及访问数组的元素

[NumPy 中创建数组及访问数组的元素](Code/1-NumPy/NumPy中创建数组及访问数组的元素.ipynb)  

#### 2.1.1 创建数组

```python
import numpy as np

# 创建一维数组
# 通过 python list 创建
list_1 = [1, 2, 3, 4] # [1, 2, 3, 4]
array_1 = np.array(list_1) # array([1, 2, 3, 4])

# 创建二维数组
list_2 = [5, 6, 7, 8]
array_2 = np.array([list_1, list_2])
# array([[1, 2, 3, 4],
#      [5, 6, 7, 8]])

# array_1 是一个一维数组
# array_2 是一个二维数组

# 其他创建数组的方法
# 快速创建一个 1-10（不含10）的数组，间隔默认为1
array_4 = np.arange(1, 10) # array([1, 2, 3, 4, 5, 6, 7, 8, 9])

# 参数依次为：起始值（包含），结束值（不包含），间隔
array_4 = np.arange(1, 10, 2) # array([1, 3, 5, 7, 9])

# 快速创建数组
# 创建一个随机元素的 10 位一维数组（元素符合正态分布）
np.random.randn(10)

# 返回一个 10 以内的 int 型 随机数
np.random.randint(10)

# 返回一个 10 以内的 int 型 2 乘 3 的二维数组
np.random.randint(10, size=(2, 3))

# 返回一个 10 以内的 int 型 20 位的一维数组
np.random.randint(10, size=20)

# 将一个 10 以内的 int 型 20 位的一维数组，转换为一个 4 乘 5 的二维数组
np.random.randint(10, size=20).reshape(4, 5)
```

#### 2.1.2 访问数组元素

```python
import numpy as np


# 获取数组属性

# array_2
# array([[1, 2, 3, 4],
#      [5, 6, 7, 8]])

# 获取元祖
array_2.shape # (2, 4) 表示为一个 2 行 4 列的数组

# 获取数组中元素的个数
array_2.size # 8

# 返回数组中元素的数据类型
# 当数组中元素的数据类型不一致时会返回范围最广的那个
array_2.dtype # dtype('int64')

# a
# array([1, 2, 3, 4, 5, 6, 7, 8, 9])

# a 的第 0 位
a[0] # 1

# a 的 第 1 位到第 4 位
a[1:5] # array([2, 3, 4, 5])

# b
# array([[1, 2, 3],
#        [4, 5, 6]])

# b 的第 1 行第 0 个元素
b[1][0] # 4

# b 的第 1 行第 0 个元素
b[1, 0] # 4 等同于 b[1][0]

# c
# array([[1, 2, 3],
#        [4, 5, 6],
#        [7, 8, 9]])

# 取 c 的第 0、1 行第 1、2 列
c[:2 ,1:]
# array([[2, 3],
#        [5, 6]])

# c
# array([[1, 2, 2, 6, 8],
#        [0, 4, 4, 6, 8],
#        [7, 8, 1, 0, 8],
#        [7, 2, 4, 0, 5]])

# 返回 c 中 唯一的（不重复）的值
np.unique(c) # array([0, 1, 2, 4, 5, 6, 7, 8])

# 求和，多维数组所有列的和
sum(c) # array([15, 16, 11, 12, 29])

# 求和，多维数组第 0 行的和
sum(c[0]) # 19

# 求和，多维数组第 0 列的和
sum(c[:, 0]) # 15

# 数组中的最大值
c.max() # 8
```

#### 2.1.3 创建矩阵

```python
import numpy as np

# 创建特殊矩阵

# 创建一维全0矩阵，5 为位数
np.zeros(5) # array([0., 0., 0., 0., 0.])

# 创建二维矩阵全0矩阵
np.zeros([2, 3]) 
# array([[0., 0., 0.],
#        [0., 0., 0.]])

# 创建单位矩阵 5 乘 5
np.eye(5) 
# array([[1., 0., 0., 0., 0.],
#        [0., 1., 0., 0., 0.],
#        [0., 0., 1., 0., 0.],
#        [0., 0., 0., 1., 0.],
#        [0., 0., 0., 0., 1.]])

# 创建矩阵

# np.mat([[1, 2, 3], [4, 5, 6]])
# matrix([[1, 2, 3],
#         [4, 5, 6]])

# 将数组 a 转换成矩阵（数组可转换为矩阵）
np.mat(a)
```

### 2.2 数组和矩阵运算

[数组和矩阵运算](Code/1-NumPy/数组和矩阵运算.ipynb)

#### 2.2.1 数组的运算

```python
import numpy as np

# 数组的运算

# a
# array([[6, 5, 5, 8, 4],
#        [8, 4, 6, 8, 1],
#        [5, 7, 1, 5, 8],
#        [0, 5, 4, 1, 2]])
# b
# array([[3, 8, 8, 2, 6],
#        [8, 1, 2, 5, 1],
#        [4, 4, 3, 1, 7],
#        [4, 8, 4, 1, 3]])

# 数组的加法
# 对应元素的和
a + b
# array([[ 9, 13, 13, 10, 10],
#        [16,  5,  8, 13,  2],
#        [ 9, 11,  4,  6, 15],
#        [ 4, 13,  8,  2,  5]])

# 数组的减法
# 对应元素的差
a - b
# array([[ 3, -3, -3,  6, -2],
#        [ 0,  3,  4,  3,  0],
#        [ 1,  3, -2,  4,  1],
#        [-4, -3,  0,  0, -1]])

# 数组的乘法
# 对应元素的积
a * b
# array([[18, 40, 40, 16, 24],
#        [64,  4, 12, 40,  1],
#        [20, 28,  3,  5, 56],
#        [ 0, 40, 16,  1,  6]])

# 数组的除法
# 对应元素的商（当除数为 0 时，会有警告，结果以 inf 表示）
a / b
# array([[2.        , 0.625     , 0.625     , 4.        , 0.66666667],
#        [1.        , 4.        , 3.        , 1.6       , 1.        ],
#        [1.25      , 1.75      , 0.33333333, 5.        , 1.14285714],
#        [0.        , 0.625     , 1.        , 1.        , 0.66666667]])
```

#### 2.2.2 矩阵的运算

```python
import numpy as np

# 矩阵的运算

# A
# matrix([[6, 5, 5, 8, 4],
#         [8, 4, 6, 8, 1],
#         [5, 7, 1, 5, 8],
#         [0, 5, 4, 1, 2]])
# B
# matrix([[3, 8, 8, 2, 6],
#         [8, 1, 2, 5, 1],
#         [4, 4, 3, 1, 7],
#         [4, 8, 4, 1, 3]])

# 矩阵的加法
# 矩阵的加法为对应元素的和
A + B
# matrix([[ 9, 13, 13, 10, 10],
#         [16,  5,  8, 13,  2],
#         [ 9, 11,  4,  6, 15],
#         [ 4, 13,  8,  2,  5]])

# 矩阵的减法
# 矩阵的减法为对应元素的差
A - B
# matrix([[ 3, -3, -3,  6, -2],
#         [ 0,  3,  4,  3,  0],
#         [ 1,  3, -2,  4,  1],
#         [-4, -3,  0,  0, -1]])

# 矩阵的乘法
# 会报错
A * B

--------
A = np.mat(np.random.randint(10, size=20).reshape(4, 5))
# A （A 为 4 乘 5 的矩阵）
# matrix([[9, 0, 3, 8, 6],
#         [4, 0, 7, 1, 8],
#         [4, 7, 2, 2, 7],
#         [2, 4, 2, 1, 5]])
B = np.mat(np.random.randint(10, size=20).reshape(5, 4))
# B （B 为 5 乘 4 的矩阵）
# matrix([[2, 0, 7, 0],
#         [4, 7, 3, 6],
#         [9, 9, 0, 9],
#         [7, 4, 9, 3],
#         [2, 1, 4, 4]])

# 4 乘 5 的矩阵与 5 乘 4 的矩阵的积为一个 4 乘 4 的矩阵 
A * B
# matrix([[113,  65, 159,  75],
#         [ 94,  75,  69,  98],
#         [ 82,  82,  95,  94],
#         [ 55,  55,  55,  65]])


```

### 2.3 Array 的 input 和 output

[Array 的 input 和 output](Code/1-NumPy/Array的input和output.ipynb)

#### 2.3.1 使用 pickle 序列化 numpy array

```python
import pickle
import numpy as np

# 查看当前目录的文件
!ls

# 使用 pickle 序列化 numpy array
# x
# array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# 使用 pickle 序列化
f = open('x.pki', 'wb') # 写二进制文件
pickle.dump(x, f)

f = open('x.pki', 'rb') # 读二进制文件
pickle.load(f)

```

#### 2.3.2 使用 numpy 序列化 numpy array

```python
import numpy as np

# 使用 numpy 序列化 numpy array
np.save('one_array', x) # 写二进制文件
np.load('one_array.npy') # 读二进制文件

# 同时序列化多个数组
np.savez('two_array.npz', a=x, b=y) # x、y 均为数组，a、b 分别为 x、y 的别名

# 读多个序列化数组的文件
c = np.load('two_array.npz')
c['a'] # 读 a
c['b'] # 读 b
```

## 3. pandas 入门

pandas 中两种重要的数据结构，此处可参考我的另一个库：[PandasVersusExcel(Python)](https://chanmenglin.github.io/PandasVersusExcel/) | [Github](https://github.com/ChanMenglin/PandasVersusExcel)

## 4. pandas 玩转数据

pandas 进阶，学习数据清洗、排序、采样技术、封箱技术、Group By、聚合技术、透视表等

## 5. 绘图与可视化-MatPlotLib

数据展示

## 6. 绘图与可视化-SeaBorn

数据展示

## 7. 数据分析项目实战

以股票市场为例从 数据获取-导入-分析-可视化 完整流程

---

# 附录

## 1. 数据科学领域5个组建 Python 库

[NumPy](http://www.numpy.org)  

* N维数组（矩阵）：快速高效、矢量数学运算
* 高效的 Index，不需要循环
* 开源免费，运行效率高（基于 C 语言），可媲美 [Matlab](https://ww2.mathworks.cn/products/matlab.html)

[SciPy](https://www.scipy.org)  

* 依赖于 NumPy
* 专为科学和工程设计
* 实现了多种常用科学计算：[线性代数](https://zh.wikipedia.org/wiki/%E7%BA%BF%E6%80%A7%E4%BB%A3%E6%95%B0)、[傅里叶变换](https://zh.wikipedia.org/wiki/%E5%82%85%E9%87%8C%E5%8F%B6%E5%8F%98%E6%8D%A2)、信号和图像处理

[pandas](https://pandas.pydata.org)  

* 结构化数据分析利器（依赖 NumPy）
* 提供多种高级数据结构：Time-Series(时间序列)、DataFrame、Panel
* 强大的数据索引和处理能力

[MatPlotLib](https://matplotlib.org)  

* Python 2D绘图领域使用最广泛的套件
* 基本能取代 Matlab 的绘图功能（柱形图、散点图、曲线图等）
* 通过 mplot3d 可绘制精美的3D图

[SciKit-learn](http://scikit-learn.org/)  

* 机器学习 Python 模块
* 建立在 SciPy 之上，提供了常用的机器学习算法：[聚类](https://zh.wikipedia.org/wiki/%E8%81%9A%E7%B1%BB%E5%88%86%E6%9E%90)、[回归](https://zh.wikipedia.org/wiki/%E8%BF%B4%E6%AD%B8%E5%88%86%E6%9E%90)
* 简单浴血的API接口

## 2. 数学基础-[矩阵运算](https://zh.wikipedia.org/wiki/%E7%9F%A9%E9%98%B5)

### 2.1 基本概念

* 矩阵：矩形的数组，即二维数组。其中向量和标量都是矩阵的特点
* 向量：是指 1×n 或者 n×1 的矩阵
* 标量：1×1 的矩阵
* 数组：N味的数组，是矩阵的延伸

### 2.2 特殊的矩阵

* 全0全1矩阵：矩阵中所有的元素都是0或都是1
* 单位矩阵
    * 为 n×n 的矩阵
    * 对角线上的元素都是1
    * 任何矩阵乘以单位矩阵，结果为原先的矩阵（类似于任何数乘以1）

![特殊的矩阵](./img/JZ01.png)

### 2.3 矩阵的加减运算

* 相加、相减的矩阵必须要有相同的列数和行数
* 行和列对应元素相加减

![矩阵的加减运算](./img/JZ02.png)

### 2.4 数组的乘法（点乘）

* 数组乘法（点乘）是对应元素之间的乘法（类似矩阵的加减法）

![数组的乘法（点乘）](./img/JZ03.png)

### 矩阵的乘法

* 设 A 为 m×p 的矩阵，B 为 p×m 的矩阵，m×n 矩阵 C 为 A 与 B 的乘积，记为 C=AB，其中矩阵 C 的第 i 行第 j 列元素可标示为：

![矩阵的乘法](./img/JZ04.png)

参考：清华大学出版社的线性代数 http://bs.szu.edu.cn/sljr/Up/day_110824/201108240409437707.pdf
