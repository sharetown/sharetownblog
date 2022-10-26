---
title: "EChats组件的使用"
description: 第三顿
date: 2022-10-26T22:26:23+08:00
draft: false

categories:
- 数据可视化

tags:
- 前端
- EChats
---

### Echats插件的使用
{{< figure src="image-20200704160442667.png" alt="image-20200704160442667" >}}
#### 一、什么是Echats？

ECharts 是一个使用 JavaScript 实现的开源可视化库，涵盖各行业图表，满足各种需求。

ECharts 遵循 Apache-2.0 开源协议，免费商用。

ECharts 兼容当前绝大部分浏览器（IE8/9/10/11，Chrome，Firefox，Safari等）及兼容多种设备，可随时随地任性展示。

#### 二、Echats配置语法

##### 第一步：创建HTML页面

创建一个HTMl页面，引入Echats的js文件

引入Echats文件有三种方法，分别是：

- 下载Echats独立版本，然后在`<script>`标签中引入`echarts.min.js`即可。`echarts.min.js`文件下载地址：`https://github.com/apache/incubator-echarts/tree/4.8.0/dist`使用Git下载


- 使用CDN方法：
  **Staticfile CDN（国内）** : https://cdn.staticfile.org/echarts/4.3.0/echarts.min.js
  **百度**：https://echarts.baidu.com/dist/echarts.min.js, 保持了最新版本。
  **cdnjs** : https://cdnjs.cloudflare.com/ajax/libs/echarts/4.3.0/echarts.min.js

  使用方法：以**Staticfile CDN**为例：

```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>第一个 ECharts 实例</title>
    <!-- 引入 echarts.js -->
    <script src="https://cdn.staticfile.org/echarts/4.3.0/echarts.min.js">
    <!-- 在head中引入js地址即可 -->
    </script>
</head>
<body>
    
</body>
</html>
```

- NPM方法（不推荐）

##### 第二步: 为 ECharts 准备一个具备高宽的 DOM 容器

```html
<body>
    <!-- 为 ECharts 准备一个具备大小（宽高）的 DOM -->
    <div id="main" style="width: 600px;height:400px;"></div>
</body>
```

##### 第三步：设置配置信息

Echats库使用json格式来配置

```html
echarts.init(document.getElementById('main')).setOption(option);
<!--这里 option 表示使用 json 数据格式的配置来绘制图表。步骤如下：-->

<!--为图表配置标题-->
title:{
	text:'标题'
}

<!--配置提示信息-->
tooltip: {},

<!--图例组件-->
legend: {
    data: [{
        name: '系列1',
        // 强制设置图形为圆。
        icon: 'circle',
        // 设置文本为红色
        textStyle: {
            color: 'red'
        }
    }]
}

<!--配置要在X轴显示的项-->
xAxis: {
    data: ["衬衫","羊毛衫","雪纺衫","裤子","高跟鞋","袜子"]
}

<!--配置要在Y轴显示的项-->
yAxis: {}

<!--选择用什么类型的图表来显示数据,在Echats中图表类型称为系列-->
series: [{
    name: '销量',  // 系列名称
    type: 'bar',  // 系列图表类型
    data: [5, 20, 36, 10, 10, 20]  // 系列中的数据内容
}]
```

**附：每个系列的表示方法**

| 图表类型 | 系列表示法 |      | 图表类型 | 系列表示法 |
| :--------: | :----------: | ---- | :--------: | :--------: |
| 柱状/条形图 | `type:'bar'` |      | 折线/面积图 | `type:'line'` |
| 饼图 | `type:'pie'` | | 散点（气泡）图 | `type:'scatter'` |
| 带有涟漪特效动画的散点（气泡） | `type:'effectScatter'` | | 雷达图 | `type: 'radar'` |
| 树型图 | `type: 'tree'` | | 树型图 | `type: 'treemap'` |
| 旭日图 | `type: 'sunburst'` | | 箱形图 | `type: 'boxplot'` |
| K线图 | `type: 'candlestick'` | | 热力图 | `type: 'heatmap'` |
| 地图 | `type: 'map'` | | 平行坐标系的系列 | `type: 'parallel'` |
| 线图 | `type: 'lines'` | | 关系图 | `type: 'graph'` |
| 桑基图 | `type: 'sankey'` | | 漏斗图 | `type: 'funnel'` |
| 仪表盘` |    `type: 'gauge'`     | | 象形柱图 | `type: 'pictorialBar'` |
| 主题河流 | `type: 'themeRiver'` | | 自定义系列 | `type: 'custom'` |

示例：

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>我的第一个Echats实例</title>
    <!-- 引入 echarts.js -->
    <script src="js/echarts.min.js"></script>
</head>
<body>
<!-- 为ECharts准备一个具备大小（宽高）的Dom -->
<div id="main" style="width: 600px;height:400px;"></div>
<script type="text/javascript">
    // 基于准备好的dom，初始化echarts实例
    var myChart = echarts.init(document.getElementById('main'));

    // 指定图表的配置项和数据
    var option = {
        title: {
            text: '第一个 ECharts 实例'
        },
        tooltip: {},
        legend: {
            data:['销量']
        },
        xAxis: {
            data: ["衬衫","羊毛衫","雪纺衫","裤子","高跟鞋","袜子"]
        },
        yAxis: {},
        series: [{
            name: '销量',
            type: 'bar',
            data: [5, 20, 36, 10, 10, 20]
        }]
    };

    // 使用刚指定的配置项和数据显示图表。
    myChart.setOption(option);
</script>
</body>
</html>
```

{{< figure src="image-20200704165605434.png" alt="image-20200704165605434" >}}

#### 三、饼图

前面的章节我们已经学会了使用 ECharts 绘制一个简单的柱状图，本章节我们将绘制饼图。

饼图主要是通过扇形的弧度表现不同类目的数据在总和中的占比，它的数据格式比柱状图更简单，只有一维的数值，不需要给类目。因为不在直角坐标系上，所以也不需要 xAxis，yAxis。

实例：

```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>饼图</title>
    <!-- 引入 echarts.js -->
    <script src="https://cdn.staticfile.org/echarts/4.3.0/echarts.min.js"></script>
</head>
<body>
<!-- 为ECharts准备一个具备大小（宽高）的Dom -->
<div id="main" style="width: 600px;height:400px;"></div>
<script type="text/javascript">
    // 基于准备好的dom，初始化echarts实例
    var myChart = echarts.init(document.getElementById('main'));

    myChart.setOption({
        title:{
            text:'饼图'
        },
        tooltip: {},
        legend: {
            data:['视频广告','联盟广告','邮件营销','直接访问','搜索引擎']
        },
        series : [
            {
                name: '访问来源',
                type: 'pie',    // 设置图表类型为饼图
                radius: '55%',  // 饼图的半径，外半径为可视区尺寸（容器高宽中较小一项）的 55% 长度。
                data:[          // 数据数组，name 为数据项名称，value 为数据项值
                    {value:235, name:'视频广告'},
                    {value:274, name:'联盟广告'},
                    {value:310, name:'邮件营销'},
                    {value:335, name:'直接访问'},
                    {value:400, name:'搜索引擎'}
                ]
            }
        ]
    })
</script>
</body>
</html>
```

{{< figure src="image-20200704165623521.png" alt="image-20200704165623521" >}}

##### `roseType: 'angle'`属性

我们也可以通过设置参数 **roseType: 'angle'** 把饼图显示成南丁格尔图

```html
series : [
            {
                name: '访问来源',
                type: 'pie',    // 设置图表类型为饼图
                radius: '55%',  // 饼图的半径，外半径为可视区尺寸（容器高宽中较小一项）的 55% 长度。

				roseType: 'angle',//把饼图显示成南丁格尔图

                data:[          // 数据数组，name 为数据项名称，value 为数据项值
                    {value:235, name:'视频广告'},
                    {value:274, name:'联盟广告'},
                    {value:310, name:'邮件营销'},
                    {value:335, name:'直接访问'},
                    {value:400, name:'搜索引擎'}
                ]
            }
        ]
```

{{< figure src="image-20200704170011618.png" alt="image-20200704170011618" >}}

##### 阴影的配置

`itemStyle `参数可以设置诸如阴影、透明度、颜色、边框颜色、边框宽度等：

```html
itemStyle: {
    normal: {
        shadowBlur: 200,
        shadowColor: 'rgba(0, 0, 0, 0.5)'
    }
}
//以上属性都是嵌入到series中
```
{{< figure src="image-20200704170504890.png" alt="image-20200704170504890" >}}

##### 高亮样式配置：`emphasis`

```html
// 高亮样式。
emphasis: {
    itemStyle: {
        // 高亮时点的颜色
        color: 'red'
    },
    label: {
        show: true,
        // 高亮时标签的文字
        formatter: '高亮时显示的标签内容'
    }
},
//以上属性都是嵌入到series中
```

#### 四、Echats异步加载数据

ECharts 通常数据设置在 setOption 中，如果我们需要异步加载数据，可以配合 jQuery等工具，在异步获取数据后通过 setOption 填入数据和配置项就行。

ECharts 通常数据设置在 setOption 中，如果我们需要异步加载数据，可以配合 jQuery等工具，在异步获取数据后通过 setOption 填入数据和配置项就行。 json 数据：

##### 数据的动态更新

ECharts 由数据驱动，数据的改变驱动图表展现的改变，因此动态数据的实现也变得异常简单。

所有数据的更新都通过 setOption 实现，你只需要定时获取数据，setOption 填入数据，而不用考虑数据到底产生了那些变化，ECharts 会找到两组数据之间的差异然后通过合适的动画去表现数据的变化。

#### 五、Echats数据集

ECharts 使用 dataset 管理数据。

dataset 组件用于单独的数据集声明，从而数据可以单独管理，被多个组件复用，并且可以基于数据指定数据到视觉的映射。

下面是一个最简单的 dataset 的例子：

```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>第一个 ECharts 实例</title>
    <!-- 引入 echarts.js -->
    <script src="https://cdn.staticfile.org/echarts/4.3.0/echarts.min.js"></script>
</head>
<body>
    <!-- 为ECharts准备一个具备大小（宽高）的Dom -->
    <div id="main" style="width: 600px;height:400px;"></div>
    <script type="text/javascript">
        // 基于准备好的dom，初始化echarts实例
        var myChart = echarts.init(document.getElementById('main'));
 
        // 指定图表的配置项和数据
        var option = {
            legend: {},
            tooltip: {},
            dataset: {
                // 提供一份数据。
                source: [
                    ['product', '2015', '2016', '2017'],
                    ['Matcha Latte', 43.3, 85.8, 93.7],
                    ['Milk Tea', 83.1, 73.4, 55.1],
                    ['Cheese Cocoa', 86.4, 65.2, 82.5],
                    ['Walnut Brownie', 72.4, 53.9, 39.1]
                ]
            },
            // 声明一个 X 轴，类目轴（category）。默认情况下，类目轴对应到 dataset 第一列。
            xAxis: {type: 'category'},
            // 声明一个 Y 轴，数值轴。
            yAxis: {},
            // 声明多个 bar 系列，默认情况下，每个系列会自动对应到 dataset 的每一列。
            series: [
                {type: 'bar'},
                {type: 'bar'},
                {type: 'bar'}
            ]
        };
 
        // 使用刚指定的配置项和数据显示图表。
        myChart.setOption(option);
    </script>
</body>
</html>
```

或者也可以使用常见的对象数组的格式：

```html
dataset: {
                    source: [
                        ['product', '2012', '2013', '2014', '2015'],
                        ['Matcha Latte', 41.1, 30.4, 65.1, 53.3],
                        ['Milk Tea', 86.5, 92.1, 85.7, 83.1],
                        ['Cheese Cocoa', 24.1, 67.2, 79.5, 86.4]
                    ]
                },
```

##### 数据到图形的映射

我们可以在配置项中将数据映射到图形中。

我么可以使用 series.seriesLayoutBy 属性来配置 dataset 是列（column）还是行（row）映射为图形系列（series），默认是按照列（column）来映射。

以下实例我们将通过 seriesLayoutBy 属性来配置数据是使用列显示还是按行显示。

```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>第一个 ECharts 实例</title>
    <!-- 引入 echarts.js -->
    <script src="https://cdn.staticfile.org/echarts/4.3.0/echarts.min.js"></script>
</head>
<body>
    <!-- 为ECharts准备一个具备大小（宽高）的Dom -->
    <div id="main" style="width: 600px;height:400px;"></div>
    <script type="text/javascript">
        // 基于准备好的dom，初始化echarts实例
        var myChart = echarts.init(document.getElementById('main'));
 
        // 指定图表的配置项和数据
        var option = {
                legend: {},
                tooltip: {},
                dataset: {
                    source: [
                        ['product', '2012', '2013', '2014', '2015'],
                        ['Matcha Latte', 41.1, 30.4, 65.1, 53.3],
                        ['Milk Tea', 86.5, 92.1, 85.7, 83.1],
                        ['Cheese Cocoa', 24.1, 67.2, 79.5, 86.4]
                    ]
                },
                xAxis: [
                    {type: 'category', gridIndex: 0},
                    {type: 'category', gridIndex: 1}
                ],
                yAxis: [
                    {gridIndex: 0},
                    {gridIndex: 1}
                ],
                grid: [
                    {bottom: '55%'},
                    {top: '55%'}
                ],
                series: [
                    // 这几个系列会在第一个直角坐标系中，每个系列对应到 dataset 的每一行。
                    {type: 'bar', seriesLayoutBy: 'row'},
                    {type: 'bar', seriesLayoutBy: 'row'},
                    {type: 'bar', seriesLayoutBy: 'row'},
                    // 这几个系列会在第二个直角坐标系中，每个系列对应到 dataset 的每一列。
                    {type: 'bar', xAxisIndex: 1, yAxisIndex: 1},
                    {type: 'bar', xAxisIndex: 1, yAxisIndex: 1},
                    {type: 'bar', xAxisIndex: 1, yAxisIndex: 1},
                    {type: 'bar', xAxisIndex: 1, yAxisIndex: 1}
                ]
            }

        // 使用刚指定的配置项和数据显示图表。
        myChart.setOption(option);
    </script>
</body>
</html>
```
{{< figure src="image-20200704174809045.png" alt="image-20200704174809045" >}}


常用图表所描述的数据大部分是"二维表"结构，我们可以使用 series.encode 属性将对应的数据映射到坐标轴（如 X、Y 轴）：

```html
series: [
                    {
                        type: 'bar',
                        encode: {
                            // 将 "amount" 列映射到 X 轴。
                            x: 'amount',
                            // 将 "product" 列映射到 Y 轴。
                            y: 'product'
                        }
                    }
                ]
```
{{< figure src="image-20200704174908940.png" alt="image-20200704174908940" >}}


encode 声明的基本结构如下，其中冒号左边是坐标系、标签等特定名称，如 'x', 'y', 'tooltip' 等，冒号右边是数据中的维度名（string 格式）或者维度的序号（number 格式，从 0 开始计数），可以指定一个或多个维度（使用数组）。通常情况下，下面各种信息不需要所有的都写，按需写即可。

下面是 encode 支持的属性：

```html
// 在任何坐标系和系列中，都支持：
encode: {
    // 使用 “名为 product 的维度” 和 “名为 score 的维度” 的值在 tooltip 中显示
    tooltip: ['product', 'score']
    // 使用 “维度 1” 和 “维度 3” 的维度名连起来作为系列名。（有时候名字比较长，这可以避免在 series.name 重复输入这些名字）
    seriesName: [1, 3],
    // 表示使用 “维度2” 中的值作为 id。这在使用 setOption 动态更新数据时有用处，可以使新老数据用 id 对应起来，从而能够产生合适的数据更新动画。
    itemId: 2,
    // 指定数据项的名称使用 “维度3” 在饼图等图表中有用，可以使这个名字显示在图例（legend）中。
    itemName: 3
}

// 直角坐标系（grid/cartesian）特有的属性：
encode: {
    // 把 “维度1”、“维度5”、“名为 score 的维度” 映射到 X 轴：
    x: [1, 5, 'score'],
    // 把“维度0”映射到 Y 轴。
    y: 0
}

// 单轴（singleAxis）特有的属性：
encode: {
    single: 3
}

// 极坐标系（polar）特有的属性：
encode: {
    radius: 3,
    angle: 2
}

// 地理坐标系（geo）特有的属性：
encode: {
    lng: 3,
    lat: 2
}

// 对于一些没有坐标系的图表，例如饼图、漏斗图等，可以是：
encode: {
    value: 3
}
```

##### 视觉通道（颜色、尺寸等）的映射

我们可以使用 visualMap 组件进行视觉通道的映射。

视觉元素可以是：

- symbol: 图元的图形类别。

- symbolSize: 图元的大小。

- color: 图元的颜色。

- colorAlpha: 图元的颜色的透明度。

- opacity: 图元以及其附属物（如文字标签）的透明度。

- colorLightness: 颜色的明暗度。

- colorSaturation: 颜色的饱和度。

- colorHue: 颜色的色调。

visualMap 组件可以定义多个，从而可以同时对数据中的多个维度进行视觉映射。

```html
visualMap: {
        orient: 'horizontal',
        left: 'center',
        min: 10,
        max: 100,
        text: ['High Score', 'Low Score'],
        // Map the score column to color
        dimension: 0,
        inRange: {
            color: ['#D7DA8B', '#E15457']
        }
    },
```

##### 交互联动

以下实例多个图表共享一个 dataset，并带有联动交互：

**以下代码暂时看不懂**

```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>ECharts 实例</title>
    <!-- 引入 echarts.js -->
    <script src="js/echarts.min.js"></script>
</head>
<body>
<!-- 为ECharts准备一个具备大小（宽高）的Dom -->
<div id="main" style="width: 600px;height:400px;"></div>
<script type="text/javascript">
    // 基于准备好的dom，初始化echarts实例
    var myChart = echarts.init(document.getElementById('main'));


    setTimeout(function () {

        option = {
            legend: {},
            tooltip: {
                trigger: 'axis',
                showContent: false
            },
            dataset: {
                source: [
                    ['product', '2012', '2013', '2014', '2015', '2016', '2017'],
                    ['Matcha Latte', 41.1, 30.4, 65.1, 53.3, 83.8, 98.7],
                    ['Milk Tea', 86.5, 92.1, 85.7, 83.1, 73.4, 55.1],
                    ['Cheese Cocoa', 24.1, 67.2, 79.5, 86.4, 65.2, 82.5],
                    ['Walnut Brownie', 55.2, 67.1, 69.2, 72.4, 53.9, 39.1]
                ]
            },
            xAxis: {type: 'category'},
            yAxis: {gridIndex: 0},
            grid: {top: '55%'},
            series: [
                {type: 'line', smooth: true, seriesLayoutBy: 'row'},
                {type: 'line', smooth: true, seriesLayoutBy: 'row'},
                {type: 'line', smooth: true, seriesLayoutBy: 'row'},
                {type: 'line', smooth: true, seriesLayoutBy: 'row'},
                {
                    type: 'pie',
                    id: 'pie',
                    radius: '30%',
                    center: ['50%', '25%'],
                    label: {
                        formatter: '{b}: {@2012} ({d}%)'
                    },
                    encode: {
                        itemName: 'product',
                        value: '2012',
                        tooltip: '2012'
                    }
                }
            ]
        };

        myChart.on('updateAxisPointer', function (event) {
            var xAxisInfo = event.axesInfo[0];
            if (xAxisInfo) {
                var dimension = xAxisInfo.value + 1;
                myChart.setOption({
                    series: {
                        id: 'pie',
                        label: {
                            formatter: '{b}: {@[' + dimension + ']} ({d}%)'
                        },
                        encode: {
                            value: dimension,
                            tooltip: dimension
                        }
                    }
                });
            }
        });

        myChart.setOption(option);

    });
</script>
</body>
</html>
```

#### 六、Echats交互组件

ECharts 提供了很多交互组件：例组件 legend、标题组件 title、视觉映射组件 visualMap、数据区域缩放组件 dataZoom、时间线组件 timeline。

接下来的内容我们将介绍如何使用数据区域缩放组件 dataZoom。

##### dataZoom

dataZoom 组件可以实现通过鼠标滚轮滚动，放大缩小图表的功能。

默认情况下 dataZoom 控制 x 轴，即对 x 轴进行数据窗口缩放和数据窗口平移操作。

```html
dataZoom: [
	{   // 这个dataZoom组件，默认控制x轴。
		type: 'slider', // 这个 dataZoom 组件是 slider 型 dataZoom 组件
        start: 10,      // 左边在 10% 的位置。
        end: 60         // 右边在 60% 的位置。
    }
],
```

上面的实例只能拖动 dataZoom 组件来缩小或放大图表。如果想在坐标系内进行拖动，以及用鼠标滚轮（或移动触屏上的两指滑动）进行缩放，那么需要 再再加上一个 inside 型的 dataZoom 组件。

在以上实例基础上我们再增加` type: 'inside' `的配置信息：

```html
dataZoom: [
	{   // 这个dataZoom组件，默认控制x轴。
		type: 'slider', // 这个 dataZoom 组件是 slider 型 dataZoom 组件
		start: 10,      // 左边在 10% 的位置。
		end: 60         // 右边在 60% 的位置。
	},
	{   // 这个dataZoom组件，也控制x轴。
		type: 'inside', // 这个 dataZoom 组件是 inside 型 dataZoom 组件
		start: 10,      // 左边在 10% 的位置。
		end: 60         // 右边在 60% 的位置。
	}
             
],
```

当然我们可以通过 dataZoom.xAxisIndex 或 dataZoom.yAxisIndex 来指定 dataZoom 控制哪个或哪些数轴。

```html
dataZoom: [
    {
        type: 'slider',
        show: true,
        xAxisIndex: [0],
        start: 1,
        end: 35
    },
    {
        type: 'slider',
        show: true,
        yAxisIndex: [0],
        left: '93%',
        start: 29,
        end: 36
    },
    {
        type: 'inside',
        xAxisIndex: [0],
        start: 1,
        end: 35
    },
    {
        type: 'inside',
        yAxisIndex: [0],
        start: 29,
        end: 36
    }
],
```

#### 七、Echats响应式

ECharts 图表显示在用户指定高宽的 DOM 节点（容器）中。

有时候我们希望在 PC 和 移动设备上都能够很好的展示图表的内容，实现响应式的设计，为了解决这个问题，ECharts 完善了组件的定位设置，并且实现了类似 CSS Media Query 的自适应能力。

##### Echats组件的定位和布局

大部分『组件』和『系列』会遵循两种定位方式。

**left/right/top/bottom/width/height 定位方式**

这六个量中，每个量都可以是『绝对值』或者『百分比』或者『位置描述』。

- 绝对值

  单位是浏览器像素（px），用 number 形式书写（不写单位）。例如 {left: 23, height: 400}。

- 百分比

  表示占 DOM 容器高宽的百分之多少，用 string 形式书写。例如 {right: '30%', bottom: '40%'}。


- 位置描述

  可以设置 left: 'center'，表示水平居中。
  可以设置 top: 'middle'，表示垂直居中。

这六个量的概念，和 CSS 中六个量的概念类似：

- left：距离 DOM 容器左边界的距离。

- right：距离 DOM 容器右边界的距离。

- top：距离 DOM 容器上边界的距离。

- bottom：距离 DOM 容器下边界的距离。

- width：宽度。

- height：高度。

在横向，left、right、width 三个量中，只需两个量有值即可，因为任两个量可以决定组件的位置和大小，例如 left 和 right 或者 right 和 width 都可以决定组件的位置和大小。 纵向，top、bottom、height 三个量，和横向类同不赘述。

**center / radius 定位方式**

- center

  是一个数组，表示 [x, y]，其中，x、y可以是『绝对值』或者『百分比』，含义和前述相同。

- radius

  是一个数组，表示 [内半径, 外半径]，其中，内外半径可以是『绝对值』或者『百分比』，含义和前述相同。

  在自适应容器大小时，百分比设置是很有用的。

##### 横向（horizontal）和纵向（vertical）

ECharts的『外观狭长』型的组件（如 legend、visualMap、dataZoom、timeline等），大多提供了『横向布局』『纵向布局』的选择。例如，在细长的移动端屏幕上，可能适合使用『纵向布局』；在PC宽屏上，可能适合使用『横向布局』。

横纵向布局的设置，一般在『组件』或者『系列』的 orient 或者 layout 配置项上，设置为 'horizontal' 或者 'vertical'。