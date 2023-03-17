# 如何自定PiP包

## 安装依赖

```bash
pip install twine==4.0.2
```

## 配置

`~`目录下新建`.pypirc`文件，复制以下内容：

```bash
[distutils]
index-servers =
    pypi
    pypitest

[pypitest]
repository: https://test.pypi.org/legacy/
username: [username]
password: [password]

[pypi]
repository = https://upload.pypi.org/legacy/
username = [username]
password = [password]

```
注意上面的两个网站需要注册账户，用一样的用户名和密码就行，然后替换掉`[username]`和`[password]`

## 上传包

在`__init__.py`中更新版本号，然后：

```bash
sh setup.sh
```