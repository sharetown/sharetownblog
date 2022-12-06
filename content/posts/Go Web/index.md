---
title: "Go Web"
description: go语言web教程
date: 2022-12-06T22:07:25+08:00
draft: false

categories:
- Golang

tags:
- 语言
- Web
- 教程
---

## Go语言Web

[TOC]

### 一、Web基础

**浏览器输入一串地址后点击回车，一直到页面被顺利加载完成，这中间都做了些什么呢？**

1. 在浏览器中输入www.qq.com域名，操作系统会<mark>先检查自己本地的hosts文件</mark>是否有这个网址映射关系，如果有，就先调用这个IP地址映射，完成域名解析。
2. 如果hosts里没有这个域名的映射，则查找<mark>本地DNS解析器缓存</mark>，是否有这个网址映射关系，如果有，直接返回，完成域名解析。
3. 如果hosts与本地DNS解析器缓存都没有相应的网址映射关系，首先会找TCP/IP参数中设置的首选DNS服务器，在此我们叫它<mark>本地DNS服务器</mark>，此服务器收到查询时，如果要查询的域名，包含在本地配置区域资源中，则返回解析结果给客户机，完成域名解析，此解析具有权威性。
4. 如果要查询的域名，不由本地DNS服务器区域解析，但该服务器已<mark>缓存了此网址映射关系</mark>，则调用这个IP地址映射，完成域名解析，此解析不具有权威性。
5. 如果本地DNS服务器本地区域文件与缓存解析都失效，则根据<mark>本地DNS服务器的设置（是否设置转发器）</mark>进行查询，如果<mark>未用转发模式</mark>，本地DNS就把请求发至 “根DNS服务器”，“根DNS服务器”收到请求后会判断这个域名(.com)是谁来授权管理，并会返回一个负责该顶级域名服务器的一个IP。本地DNS服务器收到IP信息后，将会联系负责.com域的这台服务器。这台负责.com域的服务器收到请求后，如果自己无法解析，它就会找一个管理.com域的下一级DNS服务器地址(qq.com)给本地DNS服务器。当本地DNS服务器收到这个地址后，就会找qq.com域服务器，重复上面的动作，进行查询，直至找到www.qq.com主机。
6. 如果用的是<mark>转发模式</mark>，此DNS服务器就会<mark>把请求转发至上一级DNS服务器</mark>，由上一级服务器进行解析，上一级服务器如果不能解析，或找根DNS或把转请求转至上上级，以此循环。不管本地DNS服务器用的是转发，还是根提示，最后都是把结果返回给本地DNS服务器，由此DNS服务器再返回给客户机。

{{< figure src="image-20221204133425659.png" alt="image-20221204133425659" >}}

> 所谓 `递归查询过程` 就是 “查询的递交者” 更替, 而 `迭代查询过程` 则是 “查询的递交者”不变。
>
> 举个例子来说，你想知道某个一起上法律课的女孩的电话，并且你偷偷拍了她的照片，回到寝室告诉一个很仗义的哥们儿，这个哥们儿二话没说，拍着胸脯告诉你，甭急，我替你查(此处完成了一次递归查询，即，问询者的角色更替)。然后他拿着照片问了学院大四学长，学长告诉他，这姑娘是xx系的；然后这哥们儿马不停蹄又问了xx系的办公室主任助理同学，助理同学说是xx系yy班的，然后很仗义的哥们儿去xx系yy班的班长那里取到了该女孩儿电话。(此处完成若干次迭代查询，即，问询者角色不变，但反复更替问询对象)最后，他把号码交到了你手里。完成整个查询过程。

通过上面的步骤，我们最后获取的是IP地址，也就是浏览器最后发起请求的时候是基于IP来和服务器做信息交互的。

#### 1、创建简单服务器端

```go
package main

import (
	"fmt"
	"log"
	"net/http"
)

func sayHello(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "Hello") //这个写入到 w 的是输出到客户端的
	w.Write([]byte(" World！"))
}

func main() {
	http.HandleFunc("/", sayHello)           //设置访问的路由
	err := http.ListenAndServe(":8080", nil) //设置监听的端口
	if err != nil {
		log.Fatal("ListenAndServe: ", err)
	}
}
```

如果要创建一个Web服务器端，则需要：

1. 调用`http.HandleFunc()`函数
2. 调用`http.ListenAndServe()`函数

#### 2、创建简单客户端

`Client`结构体的`Get()`和`Post()`函数直接使用了`NewRequest()`函数。`NewRequest()`函数是一个通用函数，其定义如下：

```go
func NewRequest(method,url string,body io.Reader)(*Request,error)
```

其中第一个参数为请求类型，比如“GET”、“POST”、“PUT”、“DELETE”等，第二个参数为请求地址。如果`body`参数实现了`io.Closer`接口，则`Request`返回值的`Body`字段会被设置为`body`参数的值，并会被`Client`结构体的`Do()`、`POST ()`和`PostForm()`方法关闭。

<mark>`Get()`、`Post()`函数的本质是：Go程序在底层传递相应的参数去调用`NewRequest()`函数。所以，在Go语言中创建客户端最核心的HTTP请求方法就是`NewRequest()`函数。因为PUT、DELETE方法在Go语言中没有单独封装，但是可以通过直接调用`NewRequest()`函数来实现。</mark>

1. 创建GET请求

   ```go
   package main
   
   import(
   	"fmt"
       "io/ioutil"
       "net/http"
   )
   
   func main(){
       resp,err := http.Get("https://www.baidu.com")
       if err != nil{
           fmt.Print("err",err)
       }
       closer := resp.Body
       bytes,err := ioutil.ReadAll(closer)
       fmt.Println(string(bytes))
   }
   ```

2. 创建POST请求

   ```go
   package main
   
   import(
       "bytes"
   	"fmt"
       "io/ioutil"
       "net/http"
   )
   
   func main(){
       url := "https://www.shirdon.com/comment/add"
       body := "{\"userId\":1,\"articleId\":1,\"comment\":\"这是一条评论\"}"
       resp,err := http.Post(url,"·application/x-www-form-urlencoded",bytes.NewBuffer([]byte(body)))
       if err != nil{
           fmt.Print("err",err)
       }
       closer := resp.Body
       bytes,err := ioutil.ReadAll(closer)
       fmt.Println(string(bytes))
   }
   ```

3. 创建PUT请求

   ```go
   package main
   
   import(
   	"fmt"
       "io/ioutil"
       "net/http"
       "strings"
   )
   
   func main(){
       url := "https://www.shirdon.com/comment/update"
       payload := strings.NewReader("{\"userId\":1,\"articleId\":1,\"comment\":\"这是一条评论\"}")
       req,_ := http.NewRequest("PUT",url,payload)
       req.Header.Add("Content-Type","application/json")
       resp,_ := http.DefaultClient.DO(req)
       
       defer resp.Body.Close()
   
       closer := resp.Body
       bytes,_ := ioutil.ReadAll(closer)
       fmt.Println(resp)
       fmt.Println(string(bytes))
   }
   ```

4. 创建DELETE请求

   ```go
   req,_ := http.NewRequest("DELETE",url,payload)
   ```

#### 3、模板引擎

[go模板引擎](https://blog.csdn.net/sumatch/article/details/117567070)

### 二、Go接收和处理Web请求

Go Web服务器请求和响应的流程：
{{< figure src="微信截图_20221206221716.png" alt="微信截图_20221206221716" >}}
1. 客户端发送请求；
2. 服务器端的多路复用器收到请求；
3. 多路复用器根据请求URL找到注册的处理器，将请求交由处理器处理；
4. 处理器执行程序逻辑，如果必要，则与数据库进行交互，得到处理结果；
5. 处理器调用模板引擎将指定的模板和上一步得到的结果渲染成客户端可识别的数据格式HTML；
6. 服务端将数据通过HTTP响应返回给客户端；
7. 客户端拿到数据，执行对应的操作（渲染呈现）

#### 1、接收请求

#### 2、处理请求

#### 3、session和cookie

### 三、Go访问数据库

### 四、Go高级网络编程

### 五、RESTful API 