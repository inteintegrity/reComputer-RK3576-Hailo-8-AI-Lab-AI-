# reComputer-RK3576-Hailo-8-AI-Lab-AI-
reComputer RK3576 + Hailo-8：基于 AI Lab 平台的边缘 AI 部署指南
# 概述
reComputer RK3576 是一款基于瑞芯微 RK3576 处理器的边缘 AI 计算设备，内置 6 TOPS NPU。通过 M.2 接口集成 Hailo-8 AI 加速模块后，总 AI 算力可扩展至 26 TOPS。

本指南将带你通过 Seeed Studio 的 reComputer AI Lab 平台，在 reComputer RK3576 + Hailo-8 组合上一站式部署主流开源 AI 模型，包括：
* 目标检测（YOLO11）
* 大模型对话（DeepSeek-R1）
* 视觉对话（Qwen2.5-VL）

全部模型在本地设备运行即可

## 性能概述

| 指标                          | 数据              |
| ----------------------------- | ----------------- |
| 目标检测延迟（RK3576）        | ~30ms/帧<sup>33</sup> |
| Hailo-8 算力                  | 26 TOPS<sup>9</sup>   |
| YOLO11n 推理性能              | 200+ FPS<sup>9</sup>  |

# 硬件准备
你需要准备以下硬件：
* reComputer RK3576-20 主机 × 1
* Hailo-8 M.2 2280 AI 加速模块 × 1
* 9-19V/3A DC × 1
* 显示器、键盘鼠标 × 1

## Hailo-8 安装步骤 
reComputer RK3576 和 Hailo-8 示意图如下图所示，其中左边为 reComputer RK3576，右边为 Hailo-8。
![硬件实拍图](./image/RK3576+Hailo-8.jpg)

具体安装步骤为：
- **步骤1.** 断开电源，打开 reComputer RK3576 外壳，图中红色框框就是 RK3576 的 M.2 M-Key 接口。
![硬件实拍图](./image/2.jpg)
- **步骤2.** 将 Hailo-8 模块插入主板的 M.2 M-Key 接口
![硬件实拍图](./image/3.jpg)
- **步骤3.** 固定螺丝，重新组装外壳。
- **步骤4.** 连接电源、网线和必要外设，启动设备。

# AI Lab 平台
reComputer AI Lab 是 Seeed Studio 为 reComputer RK3576/RK3588 打造的一站式 AI 模型库。其核心特点包括：
* 一键部署：选择一个预设，系统自动下载 Docker 镜像（1–4GB）并加载模型
* 三套模型预设：目标检测、大模型对话、视觉对话，相互独立可同时部署
* 标准 API 接口：REST + OpenAI 兼容接口，现有客户端可直接接入
* NPU 加速：充分利用瑞芯微 NPU 和 Hailo-8 进行高效推理

## 支持的预设与模型

| 预设     | 模型                          | 适用场景                                   |
| -------- | ----------------------------- | ------------------------------------------ |
| 目标检测 | YOLO11n（极速/均衡/精确版）    | 图片或实时视频中识别 80 种常见物体         |
| 大模型对话 | DeepSeek-R1（1.5B / 7B）     | OpenAI 兼容的本地对话 API                  |
| 视觉对话 | Qwen2.5-VL                    | 看图说话、对图片提问                       |

# reComputer RK3576 + Hailo-8 软件包部署指南

- **步骤1.** 进入hailo官网，在搜索栏搜索Software Downloads
![硬件实拍图](./image/进入hailo社区下载文件.png)
- **步骤2.** 选择对应的硬件，这里选择Hailo-8
![硬件实拍图](./image/hailo型号选择.png)
- **步骤3.** 具体文件下载
![硬件实拍图](./image/安装包选择.png)

### Windows PowerShell 上传deb包到RK3576
在Windows终端(PowerShell/Windows Terminal)执行：


scp C:\Users\seeed\Downloads\hailort-pcie-driver_4.24.0_all.deb seeed@192.168.10.230:/home/seeed/
scp C:\Users\seeed\Downloads\hailort-4.24.0-cp311-cp311-linux_aarch64.whl seeed@192.168.10.230:/home/seeed/
scp C:\Users\seeed\Downloads\librknnrt.so seeed@192.168.10.230:/home/seeed/

然后在RK3576终端执行：

```bash
# Install the PCIe driver
sudo dpkg -i hailort-pcie-driver_4.24.0_all.deb

# Reboot the system
sudo reboot

# After reboot, verify the driver is loaded
lsmod | grep hailo

# Install HailoRT
sudo dpkg -i hailort_4.24.0_arm64.deb

# Scan and verify device status
hailortcli scan

# Create and activate a virtual environment
python3 -m venv hailo_env
source hailo_env/bin/activate

# Install HailoRT Python library
pip install hailort-4.23.0-cp311-cp311-linux_aarch64.whl

# Verify installation and device connection
python3 -c "from hailo_platform import VDevice; vdev = VDevice(); print('Successfully connected via VDevice! Device info:', vdev)"
```


