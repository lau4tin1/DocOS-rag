# DocOS 安装指南

## 环境要求

安装 DocOS 之前,请确认你的系统满足以下要求:

- 操作系统:macOS 12+ / Ubuntu 20.04+ / Windows 10+
- Python 版本:3.10 或更高
- 内存:至少 8 GB
- 磁盘空间:至少 5 GB 可用空间

## 安装步骤

### 使用 pip 安装

最简单的安装方式是使用 pip,在终端中执行:

```
pip install docos
```

安装完成后,可以用下面的命令验证是否成功:

```
docos --version
```

如果输出了版本号,说明安装成功。

### 从源码安装

如果你需要最新的开发版本,可以从 GitHub 克隆源码后安装:

```
git clone https://github.com/example/docos.git
cd docos
pip install -e .
```

## 配置

首次运行前需要创建一个配置文件 `config.yaml`,其中至少包含 API 密钥:

```
api_key: your-api-key-here
```

生产环境请使用环境变量 `DOCOS_API_KEY`,不要在配置文件中硬编码密钥。

## 常见问题

### 安装时报权限错误

如果在安装时遇到 Permission denied 错误,说明当前用户没有写入权限。可以改用用户级安装:

```
pip install --user docos
```

### 找不到 docos 命令

安装成功但终端提示 command not found,通常是 PATH 没有包含 pip 的 bin 目录。请把以下目录加入 PATH:

```
~/.local/bin
```

然后重新打开终端即可。
