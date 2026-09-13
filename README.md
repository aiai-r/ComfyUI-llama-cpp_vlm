本 fork 插件有以下两个功能：

一、使用 Gemma4 GGUF 视觉模型做图片反推提示词时，可选择是否输出 think 思考内容，默认不输出。

二、输入图片不变、各节点参数不变、种子设为固定值（fixed）时，不会运行反推，直接使用之前生成的提示词。

![Gemma4 thinking controls preview](./img/gemma4-thinking-controls.png)


## **一、Gemma4 Think 思考内容控制**

本 fork 在 `Llama-cpp Model Loader` 节点增加了 `启用思考`，在 `Llama-cpp Instruct` 节点增加了 `输出think块`。

日常用 Gemma4 做图片反推提示词时，建议保持默认：

```text
启用思考: false
输出think块: false
```

这样输出内容会更像普通提示词，不会把 `<think>...</think>` 或 channel 思考内容混进提示词里。

如果你想观察模型原始输出，可以手动打开 `输出think块`。

本 fork 主要按下面这个模型测试：

[HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced](https://huggingface.co/HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced)

模型作者推荐的采样参数也已作为默认值：

```text
temperature: 0.6
top_k: 64
top_p: 0.9
min_p: 0.05
repeat_penalty: 1.1
```


## **二、固定种子缓存**

如果你把 `生成前控制` 设为 `fixed`，并且输入图片、模型、prompt、采样参数等都不变，插件会直接沿用上一次反推出来的提示词，不再重新读图反推。

这个功能适合调图时使用：先反推一次提示词，后面反复调整 KSampler 或其他节点时，就不用每次都在图片反推节点上多等十几秒。

只要你换了图片，或改了任意会影响反推结果的参数，插件会自动重新反推。


## **三、安装方法**

将本 fork 克隆到 ComfyUI 的 `custom_nodes`：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/aiai-r/ComfyUI-llama-cpp_vlm.git
```

如果你的 ComfyUI 环境已经能正常运行原版插件，一般不建议随意重装依赖。

只有在当前环境缺少依赖时，才考虑安装：

```bash
python -m pip install -r ComfyUI-llama-cpp_vlm/requirements.txt
```

安装或更新后，如果旧工作流里看不到新增选项，建议删除旧的相关节点，然后重新拉一次下述三个节点：

```text
Llama-cpp Model Loader
Llama-cpp Parameters
Llama-cpp Instruct
```


## **四、致谢**

- 原 fork：[lihaoyun6/ComfyUI-llama-cpp_vlm](https://github.com/lihaoyun6/ComfyUI-llama-cpp_vlm)
- 上游项目：[kijai/ComfyUI-llama-cpp](https://github.com/kijai/ComfyUI-llama-cpp)
- [llama-cpp-python](https://github.com/JamePeng/llama-cpp-python)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
