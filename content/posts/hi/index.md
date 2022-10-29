---
title: "如何使用Hugo引用Blowfish主题创建一个唯美的博客"
description: Blowfish非常灵活，非常适合基于静态页面的内容或带有最近帖子提要的传统博客
date: 2022-10-25T21:25:00+08:00
draft: false

categories:
- 博客

tags:
- Hugo
- Blowfish
- Markdown
---
Hugo是Go编写的静态网站生成器，速度快,易用，可配置。Hugo 有一个内容和模板目录，把他们渲染到完全的HTML网站。Hugo非常适合博客，文档等等网站的生成。Hugo依赖于Markdown文件，元数据字体。用户可以从任意的目录中运行 Hugo，支持共享主机和其他系统。

### 一、安装Hugo（windows）
- Github下载Hugo的发行版<a>https://github.com/gohugoio/hugo/releases</a>
- 本人推荐下载extended版的
{{< figure src="QQ截图20221029084734.png" alt="QQ截图20221029084734" >}}
- 将下载好的zip文件解压到你想存放的hugo的安装目录。解压后的文件夹中有三个文件，其中有一个是exe可执行文件。<mark>不要执行它，这不是安装程序</mark>。把它的路径配置到<mark>环境变量</mark>中
{{< figure src="QQ截图20221029084832.png" alt="QQ截图20221029084832" >}}
- 配置好环境变量后到<mark>命令行窗口</mark>输入`hugo version`查看是否安装成功
{{< figure src="QQ截图20221029085538.png" alt="QQ截图20221029085538" >}}
- 现在你可以通过`hugo new site YourBlogProjectName`命令来创建一个新的站点了
{{< figure src="QQ截图20221029090024.png" alt="QQ截图20221029090024" >}}
- 创建好站点后在当前界面执行`cd`命令进入到你的站点文件夹下，然后执行`hugo server`命令就可以开启这个站点了。此时访问http://localhost:1313/ 就是你站点的地址，由于是一个全新的站点此时访问是没有任何东西的。
{{< figure src="QQ截图20221029090710.png" alt="QQ截图20221029090710" >}}
至此，你的hugo以及站点就安装配置完成，下一步就是为你的站点选一个好看的主题，如果你对`Blowfish`这个主题感兴趣那就继续往下看吧
### 二、安装Blowfish
主题地址：<a>https://themes.gohugo.io/themes/blowfish/</a>
Blowfish的作者提供了三种使用该主题的方式，但是我本人比较喜欢作者提供的Hugo Module的方式，以下对该方式展开讲解。
> 作者提供的安装文档查看：https://nunocoracao.github.io/blowfish/docs/installation/#install-using-hugo
- 1、模块初始化。上面我们已经创建了一个hugo项目`myBlog`，现在我们要基于它做一个初始化。以下步骤建议到`VS Code`中进行。在VS Code中导入myBlog项目后执行`hugo mod init myBlog`。此时你会看到左边资源管理器会多出来一个`go.mod`文件
{{< figure src="QQ截图20221029092419.png" alt="QQ截图20221029092419" >}}
- 2、在myBlog项目中按如下路径新建配置文件`config/_default/module.toml`。没有文件夹需要先创建。你也可以访问这个地址去下载副本（建议）因为这里有所有会涉及到的配置文件，会简化你后面所有的操作：<a>https://minhaskamal.github.io/DownGit/#/home?url=https:%2F%2Fgithub.com%2Fnunocoracao%2Fblowfish%2Ftree%2Fmain%2Fconfig%2F_default</a>你将得到这样一个文件结构
```
config/_default/
├─ config.toml
├─ markup.toml
├─ menus.toml
├─ module.toml
└─ params.toml

```
- <mark>现在到</mark>`module.toml`<mark>中删除里面原先的所有内容，然后复制下面的内容到里面去：</mark>
```
[[imports]]
path = "github.com/nunocoracao/blowfish/v2"

```
- <mark>然后执行</mark>`hugo server`<mark>命令启动服务器，主题将自动下载</mark>
等待命令执行完，项目启动后即可访问
{{< figure src="QQ截图20221029094001.png" alt="QQ截图20221029094001" >}}
{{< figure src="QQ截图20221029094011.png" alt="QQ截图20221029094011" >}}
这便是主题的下载与安装。下面演示如何进行主题配置

### 三、配置主题
#### 1、中文
- 到`config\_default\config.toml`下设置`defaultContentLanguage = "zh-cn"`
- 再到config\_default下创建名为`languages.zh-cn.toml`的配置文件，可以复制`languages.en.toml`的来修改
- 在`languages.zh-cn.toml`中去修改你这个网站对应的中文的一些信息
#### 2、菜单
- 如果你启用的中文，需要到config\_default下新建一个名为`menus.zh-cn.toml`的配置文件，同样你可以直接复制menus.en.toml`的来修改
- `[[main]]`和`[[footer]]`两种标签，分别表示右上角和左下角的菜单项。`name`为该标签在页面上显示的名称
  `pageRef`为该标签访问的路由。
#### 3、替换网页标签图标
- 网页标签图标通常是由`<head></head>`中的`<link>`来设置，在这里我们需要新建一个html文件，在里面设置link标签来替换原生的link
- 到layouts文件夹下新建`\partials\favicons.html`目录与文件
- 到favicons.html文件中设置`<link rel="icon" type="image/jpg" href="./favicon_hp.jpg" sizes="16x16">`
- 图标文件需要存放到static目录中
#### 4、其他更多参数配置
更多的参数配置参考主题详细文档：<a>https://nunocoracao.github.io/blowfish/docs/configuration/</a>
### 四、发布文章
- 假如你菜单配置的Posts指向的是`posts`，那么你需要在content文件夹下再创建一个posts文件夹。
- 每个文章还需要创建单独的文件夹，用来存放markdown文件和图片等静态资源，加入你开启了文章缩略图，那么每个文章的markdown文件命名就都得是`index.md`而对应的缩略图呢就得在同一目录下命名为`featured.png`或`featured.jpg`。像这样：
```
content
└── posts
    └──awesome_article
       ├── index.md
       └── featured.png


```

- 头部格式：
```
---
title: "标题"
description: 描述
date: 2022-10-25T21:25:00+08:00
draft: false

categories:
- 分类

tags:
- 标签1
- 标签2
- 标签3
---
```
`date: 2022-10-25T21:25:00+08:00`时间格式不要搞错