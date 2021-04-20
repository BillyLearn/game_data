阿里云镜像



### **先创建命名空间，再创建本地镜像容器**

登录

```bash
sudo docker login --username=2929734142@qq.com registry.cn-shenzhen.aliyuncs.com
```

在命名空间那，的密码访问修改凭证密码。





从Registry中拉取镜像

```bash
sudo docker pull registry.cn-shenzhen.aliyuncs.com/bupi_tianchi/bupi_tianchi_game:[镜像版本号]
```



编辑提交docker 

```bash
sudo docker build -t registry.cn-shenzhen.aliyuncs.com/bupi_tianchi/bupi_tianchi_game:1.2 .
```



运行测试镜像

```bash
sudo nvidia-docker  run your_image(自己替换) sh run.sh
```



删除单个ID镜像，也可只输出前面三位，

```bash
sudo docker rmi eec -f
```

删除所有镜像

```bash
sudo docker image prune -a
```

