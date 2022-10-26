---
title: "Numpy的简单入门"
description: 
date: 2022-10-26T21:58:57+08:00
draft: false

categories:
- python

tags:
- 机器学习
- Numpy
---
{{<katex>}} 
NumPy是一个功能强大的Python库，主要用于对多维数组执行计算。NumPy这个词来源于两个单词-- Numerical和Python。NumPy提供了大量的库函数和操作，可以帮助程序员轻松地进行数值计算。这类数值计算广泛用于以下任务：
- 机器学习模型：在编写机器学习算法时，需要对矩阵进行各种数值计算。例如<mark>矩阵乘法、换位、加法</mark>等。NumPy提供了一个非常好的库，用于简单(在编写代码方面)和快速(在速度方面)计算。NumPy数组用于存储训练数据和机器学习模型的参数。
- 图像处理和计算机图形学：计算机中的图像表示为多维数字数组。NumPy成为同样情况下最自然的选择。实际上，NumPy提供了一些优秀的库函数来快速处理图像。例如，<mark>镜像图像、按特定角度旋转图像</mark>等。
- 数学任务：NumPy对于执行各种数学任务非常有用，如<mark>数值积分、微分、内插、外推</mark>等。因此，当涉及到数学任务时，它形成了一种基于Python的MATLAB的快速替代。

### 一、Numpy中的数组
`import numpy as np`<br>
函数|作用
--|--
`np.array()`|生成一维或多维数组
`np.shape`|查看数组的形状。
`np.zeros((n))`| 生成一个n元素的一维的全0数组
`np.zeros((n,m))`| 生成一个n*m维的全0数组
`np.ones((n))`|生成全1的数组
`np.ones((n,m))`|生成n*m的全1的数组
`np.random.random((n))`|随机生成一个n元素的一维数组
`(())`|凡是带两个括号的函数里面跟的都是向量。可以是多个数
`np.linspace(begin,end,n步数)`|这个方法是生成begin到end之间的平均步长的n个数。n就是步数
### 二、Numpy中数组的操作
#### 多维数组切片
```Python
# MD slicing
print(a[0, 1:4]) # >>>[12 13 14]
print(a[1:4, 0]) # >>>[16 21 26]
print(a[::2,::2]) # >>>[[11 13 15]
                  #     [21 23 25]
                  #     [31 33 35]]
print(a[:, 1]) # >>>[12 17 22 27 32]
```

$$
  \begin{bmatrix}
    11&12&13&14&15 \\\\
    16&17&18&19&20 \\\\
    21&22&23&24&25 \\\\
    26&27&28&29&30 \\\\
    31&32&33&34&35
  \end{bmatrix}
$$

#### 数组属性
在使用 NumPy 时，你会想知道数组的某些信息。很幸运，在这个包里边包含了很多便捷的方法，可以给你想要的信息。
```python
# Array properties
a = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28 ,29, 30],
              [31, 32, 33, 34, 35]])

print(type(a)) # >>><class 'numpy.ndarray'>
print(a.dtype) # >>>int64
print(a.size) # >>>25
print(a.shape) # >>>(5, 5)
print(a.itemsize) # >>>8
print(a.ndim) # >>>2
print(a.nbytes) # >>>200
```
正如你在上面的代码中看到的，NumPy数组实际上被称为ndarray。我不知道为什么他妈的它叫ndarray，如果有人知道请留言！我猜它代表n维数组。

数组的形状是它有多少行和列，上面的数组有5行和5列，所以它的形状是(5，5)。

itemsize属性是每个项占用的字节数。这个数组的数据类型是int 64，一个int 64中有64位，一个字节中有8位，除以64除以8，你就可以得到它占用了多少字节，在本例中是8。

ndim 属性是数组的维数。这个有2个。例如，向量只有1。

nbytes 属性是数组中的所有数据消耗掉的字节数。你应该注意到，这并不计算数组的开销，因此数组占用的实际空间将稍微大一点。
#### 基本操作符
使用NumPy，你可以轻松地在数组上执行数学运算。例如，你可以添加NumPy数组，你可以减去它们，你可以将它们相乘，甚至可以将它们分开。
以下是一些例子：
```python
import numpy as np 
a = np.array([[1.0, 2.0], [3.0, 4.0]]) 
b = np.array([[5.0, 6.0], [7.0, 8.0]]) 
sum = a + b 
difference = a - b 
product = a * b 
quotient = a / b 
print "Sum = \n", sum 
print "Difference = \n", difference 
print "Product = \n", product 
print "Quotient = \n", quotient 

# The output will be as follows: 

Sum = [[ 6. 8.] [10. 12.]]
Difference = [[-4. -4.] [-4. -4.]]
Product = [[ 5. 12.] [21. 32.]]
Quotient = [[0.2 0.33333333] [0.42857143 0.5 ]]
```
<mark>乘法运算符执行逐元素乘法而不是矩阵乘法。 要执行矩阵乘法，你可以执行以下操作：</mark>
```python
matrix_product = a.dot(b) 
print "Matrix Product = ", matrix_product
```
<mark>矩阵a乘以矩阵b表示为</mark>`a.dot(b)`
#### 布尔屏蔽
布尔屏蔽是一个有用的功能，它允许我们根据我们指定的条件检索数组中的元素。
```python
# Boolean masking
import matplotlib.pyplot as plt

a = np.linspace(0, 2 * np.pi, 50)
b = np.sin(a)
plt.plot(a,b)
mask = b >= 0
plt.plot(a[mask], b[mask], 'bo')
mask = (b >= 0) & (a <= np.pi / 2)
plt.plot(a[mask], b[mask], 'go')
plt.show()
```
#### 缺省索引
不完全索引是从多维数组的第一个维度获取索引或切片的一种方便方法。例如，如果数组a=[1，2，3，4，5]，[6，7，8，9，10]，那么[3]将在数组的第一个维度中给出索引为3的元素，这里是值4。
```python
# Incomplete Indexing
a = np.arange(0, 100, 10)
b = a[:5]
c = a[a >= 50]
print(b) # >>>[ 0 10 20 30 40]
print(c) # >>>[50 60 70 80 90]
```
#### Where函数
where() 函数是另外一个根据条件返回数组中的值的有效方法。只需要把条件传递给它，它就会返回一个使得条件为真的元素的列表。
```python
# Where
a = np.arange(0, 100, 10)
b = np.where(a < 50) 
c = np.where(a >= 50)[0]
print(b) # >>>(array([0, 1, 2, 3, 4]),)
print(c) # >>>[5 6 7 8 9]
```
### 三、创建Numpy数组的不同方式
创建Numpy数组有三种不同的方法：
1. 使用Numpy内部功能函数
1. 从列表等其他Python的结构进行转换
1. 使用特殊的库函数

#### 1、使用Numpy内部功能函数
##### 创建一个一维数组
首先，让我们创建一维数组或rank为1的数组。arange是一种广泛使用的函数，用于快速创建数组。<mark>将值20传递给arange函数会创建一个值范围为0到19的数组。</mark>
```python
import Numpy as np
array = np.arange(20)
array
```
输出
```python
array([0,  1,  2,  3,  4,
       5,  6,  7,  8,  9,
       10, 11, 12, 13, 14,
       15, 16, 17, 18, 19])
```
要<mark>验证</mark>此数组的<mark>维度</mark>，请使用<mark>shape</mark>属性。
```python
array.shape
```
输出：
```python
(20,)
```
由于逗号后面没有值，因此这是一维数组。 要访问此数组中的值，请指定非负索引。 <mark>与其他编程语言一样，索引从零开始</mark>。 因此，要访问数组中的第四个元素，请使用索引3。
```python
array[3]
```
输出：
```python
3
```
<mark>Numpy的数组是可变的</mark>，这意味着你可以在初始化数组后更改数组中元素的值。 使用print函数查看数组的内容。
```python
array[3] = 100
print(array)
```
输出：
```python
[  0   1   2 100
   4   5   6   7
   8   9  10  11
   12  13  14  15
   16  17  18  19]
```
<mark>与Python列表不同，Numpy数组的内容是同质的。 因此，如果你尝试将字符串值分配给数组中的元素，其数据类型为int，则会出现错误。</mark>
```python
array[3] ='Numpy'
```
输出：
```python
ValueError: invalid literal for int() with base 10: 'Numpy'
```
##### 创建一个二维数组
我们来谈谈创建一个二维数组。 如果只使用arange函数，它将输出一维数组。 <mark>要使其成为二维数组，请使用reshape函数链接其输出。</mark>
```pyhton
array = np.arange(20).reshape(4,5)
array
```
输出：
```python
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14],
       [15, 16, 17, 18, 19]])
```
首先，将创建20个整数，然后将数组转换为具有4行和5列的二维数组。 我们来检查一下这个数组的维数。
```python
(4, 5)
```
由于我们得到两个值，这是一个二维数组。 要访问二维数组中的元素，需要为行和列指定索引。
```python
array[3][4]
```
输出：
```python
19
```
##### 创建三维数组及更多维度
要创建三维数组，请为重塑形状函数指定3个参数。
```python
array = np.arange(27).reshape(3,3,3)
array
```
输出：
```python
array([[[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8]],

       [[ 9, 10, 11],
        [12, 13, 14],
        [15, 16, 17]],

       [[18, 19, 20],
        [21, 22, 23],
        [24, 25, 26]]])
```
需要注意的是：<mark>数组中元素的数量（27）必须是其尺寸（3 * 3 * 3）的乘积。</mark> 要交叉检查它是否是三维数组，可以使用shape属性。
```pyhton
array.shape
```
输出：
```pyhton
(3, 3, 3)
```
此外，使用arange函数，你可以创建一个在定义的起始值和结束值之间具有特定序列的数组。
```python
np.arange(10, 35, 3)
```
输出：
```python
array([10, 13, 16, 19, 22, 25, 28, 31, 34])
```
#### 2、使用其他Numpy函数
除了arange函数之外，你还可以使用其他有用的函数（如 zeros 和 ones）来快速创建和填充数组。

<mark>使用zeros函数创建一个填充零的数组</mark>。函数的参数表示行数和列数（或其维数）。
```pthon
np.zeros((2,4))
```
输出：
```pthon
array([[0., 0., 0., 0.],
       [0., 0., 0., 0.]])
```
<mark>使用ones函数创建一个填充了1的数组</mark>。
```pthon
np.ones((3,4))
```
输出：
```pthon
array([[1., 1., 1., 1.],
       [1., 1., 1., 1.],
       [1., 1., 1., 1.]])
```
<mark>empty函数创建一个数组。它的初始内容是随机的，取决于内存的状态</mark>。
```pthon
np.empty((2,3))
```
输出：
```pthon
array([[0.65670626, 0.52097334, 0.99831087],
       [0.07280136, 0.4416958 , 0.06185705]])
```
<mark>full函数创建一个填充给定值的n * n数组</mark>。
```pthon
np.full((2,2), 3)
```
输出：
```pthon
array([[3, 3],
       [3, 3]])
```
<mark>eye函数可以创建一个n * n矩阵，对角线为1，其他为0</mark>。
```pthon
np.eye(3,3)
```
输出：
```pthon
array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])
```
<mark>函数linspace在指定的时间间隔内返回均匀间隔的数字。</mark> 例如，下面的函数返回0到10之间的四个等间距数字。
```pthon
np.linspace(0, 10, num=4)
```
输出：
```pthon
array([ 0., 3.33333333, 6.66666667, 10.])
```
#### 3、从Python列表转换
除了使用Numpy函数之外，你还可以直接从Python列表创建数组。将Python列表传递给数组函数以创建Numpy数组：
```python
array = np.array([4,5,6])
array
```
输出：
```python
array([4, 5, 6])
```
你还可以创建Python列表并传递其变量名以创建Numpy数组。
```python
list = [4,5,6]
list
```
输出：
```python
[4, 5, 6]
array = np.array(list)
array
```
输出：
```python
array([4, 5, 6])
```
你可以确认变量array和list分别是Python列表和Numpy数组。
```python
type(list)
```
```python
list
```
```python
type(array)
```
```python
Numpy.ndarray
```
要创建二维数组，请将一系列列表传递给数组函数。
```python
array = np.array([(1,2,3), (4,5,6)])
array
```
输出：
```python
array([[1, 2, 3],
       [4, 5, 6]])
```
```python
array.shape
```
输出：
```python
(2, 3)
```
#### 4、使用特殊的库函数
你还可以使用特殊库函数来创建数组。例如，要创建一个填充0到1之间随机值的数组，请使用random函数。这对于需要随机状态才能开始的问题特别有用。
```python
np.random.random((2,2))
```
输出：
```python
array([[0.1632794 , 0.34567049],
       [0.03463241, 0.70687903]])
```
### 四、Numpy中的矩阵和向量
numpy的ndarray类用于表示矩阵和向量。 要在numpy中构造矩阵，我们在列表中列出矩阵的行， 并将该列表传递给numpy数组构造函数。

例如，构造与矩阵对应的numpy数组
$$
\begin{bmatrix}1&-1&2\\3&2&0\end{bmatrix}
$$
我们会这样做
```python
A = np.array([[1,-1,2],[3,2,0]])
```
向量只是具有单列的数组。 例如，构建向量
$$
\begin{bmatrix}2\\1\\3\end{bmatrix}
$$
我们会这样做
```python
v = np.array([[2],[1],[3]])
```
更方便的方法是<mark>转置相应的行向量</mark>。 例如，为了使上面的矢量，我们可以改为转置行向量
$$
\begin{bmatrix}2&1&3\end{bmatrix}
$$
这个代码是
```python
v = np.transpose(np.array([[2,1,3]]))
```
numpy重载数组索引和切片符号以访问矩阵的各个部分。 例如，要打印矩阵A中的右下方条目，我们会这样做
```python
print(A[1,2])
```
要切出A矩阵中的第二列，我们会这样做
```python
col = A[:,1:2]
```
第一个切片选择A中的所有行，而第二个切片仅选择每行中的中间条目。

<mark>要进行矩阵乘法或矩阵向量乘法，我们使用np.dot()方法。</mark>
```python
w = np.dot(A,v)
```
#### 用numpy求解方程组
线性代数中比较常见的问题之一是求解矩阵向量方程。 这是一个例子。 我们寻找解决方程的向量x

\\(Ax = b\\)

当

$$
A=\begin{bmatrix}
  2&1&-2 \\\\
  3&0&1 \\\\
  1&1&-1
 \end{bmatrix}
 \\\\
b=\begin{bmatrix}-3\\\\5\\\\-2\end{bmatrix}
$$

我们首先构建A和b的数组。
```python
A = np.array([[2,1,-2],[3,0,1],[1,1,-1]])
b = np.transpose(np.array([[-3,5,-2]]))
```
<mark>为了解决A矩阵除以b向量</mark>
```python
x = np.linalg.solve(A,b)
```
# Numpy总结
<mark>首先，python中的普通数组的类型是</mark>`list`，<mark>而Numpy中的数组（一维叫向量，二维叫矩阵）的类型都是</mark>`numpy.ndarray`
方法|解释
--|--
np.array(list)|将list数组创建成numpy中的矩阵即ndarray
np.arange(n)|创建一个一维的从0到n-1的ndarray数组
array.shape(数组变量名.shape)|查看数组的维度，几乘几
array.ndim(数组变量名.ndim)|查看数组的维度，只显示几维
array.sum(数组变量名.sum)|求和
array.max(数组变量名.max)|求最大值
array.min(数组变量名.min)|求最小值
numpy中数组元素的访问与赋值|跟普通数组一样，通过下标访问，从0开始。可直接通过=赋值
numpy可以存任意数据么|与普通数组不同，numpy只能存同类型的数据。
np.arange(n),reshape(x,y)|将np.arange(n)这个一维数组转化成x\*y的二维矩阵。x\*y必须等于n。reshape中的参数可以很多维，但是乘积必须等于n
np.zeros((n,m))|生成n*m的全0矩阵
np.ones((n,m))|生成n*m的全1矩阵
np.eye((n,m))|生成n*m的单位矩阵
np.empty((n,m))|随机生成一个n*m的矩阵，与内存状态有关，不可控
np.full((n,m),k)|生成一个全是k的n*m的矩阵
np.linspace(begin,end,num)|将begin到end分成num份，然后生成一个一维数组。闭区间
np.dot(a,b)|矩阵a乘以矩阵b，遵循高代
np.transpose(a)|转置矩阵a
np.linalg.solve(a,b)|矩阵a除以矩阵b，遵循高代