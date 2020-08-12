# StockBlog
这个股票博客系统是一个简易的预测股票日均价的查询系统,其中主要功能如下:
   * 全部股票列表
   * 股票预测详情
   * 股票编码搜索      
   
## 说明
使用的网站框架: Django  
使用的数据库: Django内置Sqlite  
使用的预测算法: LSTM  
算法中用到的包: tushare,pandas,numpy,tensorflow

## 安装
在您的系统上装好python 2.7,Django框架,以及各种各样的运算包,此处推荐使用anaconda,自带比较多python中必要的包.  
环境安好后,在命令行运行运行 python manage.py runserver localhost:"your port",即可运行.  

数据库管理员用户:admin
    密码 firstTest


## 部署
http://123.206.51.151:7777/(现已失效)
项目部署在这台服务器了.访问可点击


## 项目截图
![系统首页](https://github/julietxiao/StockBlog/raw/master/pic/home.png)
![股票预测](https://github/julietxiao/StockBlog/raw/master/pic/predict.png)
![股票列表](https://github/julietxiao/StockBlog/raw/master/pic/list.png)

