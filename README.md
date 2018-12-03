# 数据科学

> 本仓库的代码全部在 Anaconda 的 jupyter NoteBook 中编写

> 数据科学工作流
> 1. Inquire：通过数据获取什么信息，解决什么问题
> 2. Obtain：数据如何获取（[Python](https://www.python.org/)）
> 3. Scrub：数据的清理、整理和分析（[pandas](https://pandas.pydata.org)、[NumPy](http://www.numpy.org)）
> 4. Explore：展示数据、数据可视化（[MatPlotLib](https://matplotlib.org)、[Seaborn](http://seaborn.pydata.org/#)）
> 5. Model：通过数据分析和机器学习对数据进行进一步分析和训练（[SciKit-learn](http://scikit-learn.org/)、[SciPy](https://www.scipy.org)、[TensorFlow](https://tensorflow.google.cn)）
> 6. iNterpret：最终结果展示（[BOKEH](https://bokeh.pydata.org/en/latest/)、[D3.js](https://d3js.org)）

# 目录（Content）

* [1. 环境搭建](#1-环境搭建)
* [2. NumPy 入门](#2-NumPy-入门)
* [3. pandas 入门](#3-pandas-入门)
* [4. pandas 玩转数据](#4-pandas-玩转数据)
* [5. 绘图与可视化-MatPlotLib](#5-绘图与可视化-MatPlotLib)
* [6. 绘图与可视化-SeaBorn](#6-绘图与可视化-SeaBorn)
* [7. 数据分析项目实战](#7-数据分析项目实战)

---

* [附录](#附录)
    * [1. 数据可续领域5个组建 Python 库](#1-数据可续领域5个组建-Python-库)
    * [2. 数学基础-矩阵运算](#2-数学基础-矩阵运算)

---

## 1. 环境搭建

各种平台使用 [Anaconda](https://www.anaconda.com/download/#macos) 搭建数据分析开发环境

环境搭建可参考 	
Timothy老师的 [搭建开发环境，跨平台原理](http://sa.mentorx.net/course/54/task/321/show)

## 2. NumPy 入门

高效的数据结构 Array（数组）介绍，是学习其他高级数据结构的基础

### 2.1 NumPy 中创建数组及访问数组的元素

[NumPy 中创建数组及访问数组的元素](Code/1-NumPy/NumPy中创建数组及访问数组的元素.ipynb)

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

## 1. 数据可续领域5个组建 Python 库

[NumPy](http://www.numpy.org)  

* N维数组（矩阵）：快速高效、矢量数学运算
* 高效的 Index，不需要循环
* 开源免费，运行效率高（基于 C语言），可媲美 [Matlab](https://ww2.mathworks.cn/products/matlab.html)

[SciPy](https://www.scipy.org)  

* 依赖于 NumPy
* 专为科学和工程设计
* 实现了多种常用科学计算：[线性代数](https://zh.wikipedia.org/wiki/%E7%BA%BF%E6%80%A7%E4%BB%A3%E6%95%B0)、[傅里叶变换](https://zh.wikipedia.org/wiki/%E5%82%85%E9%87%8C%E5%8F%B6%E5%8F%98%E6%8D%A2)、信号和图像处理

[pandas](https://pandas.pydata.org)  

* 结构化数据分析利器（依赖 NumPy）
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
* 向量：是指 1×n 或者 n×1 的矩阵
* 标量：1×1 的矩阵
* 数组：N味的数组，是矩阵的延伸

### 2.2 特殊的矩阵

* 全0全1矩阵：矩阵中所有的元素都是0或都是1
* 单位矩阵
    * 为 n×n 的矩阵
    * 对角线上的元素都是1
    * 任何矩阵乘以单位矩阵，结果为原先的矩阵（类似于任何数乘以1）

![特殊的矩阵](./img/JZ01.png)

### 2.3 矩阵的加减运算

* 相加、相减的矩阵必须要有相同的列数和行数
* 行和列对应元素相加减

![矩阵的加减运算](./img/JZ02.png)

### 2.4 数组的乘法（点乘）

* 数组乘法（点乘）是对应元素之间的乘法（类似矩阵的加减法）

![数组的乘法（点乘）](./img/JZ03.png)

### 矩阵的乘法

* 设 A 为 m×p 的矩阵，B 为 p×m 的矩阵，m×n 矩阵 C 为 A 与 B 的乘积，记为 C=AB，其中矩阵 C 的第 i 行第 j 列元素可标示为：

![矩阵的乘法](./img/JZ04.png)

参考：清华大学出版社的线性代数(http://bs.szu.edu.cn/sljr/Up/day_110824/201108240409437707.pdf)