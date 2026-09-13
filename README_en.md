## Persistent image analysis

Enable `cache_image_analysis` on `Llama-cpp Instruct` to describe each image once and save the description under `cache/image_analysis/`. Later runs use the saved text plus your current instructions, including after a restart. Images are not attached to the final prompt-generation call. Details omitted from the description will therefore not be available to that call.

The key includes image content, model configuration, model/projector file metadata, and the analysis prompt version. Changing instructions or generation seed does not repeat image analysis. Delete the corresponding cache files and change the generation seed to refresh descriptions; disable the option to return to direct image prompting. With the option disabled (the default), existing workflows retain their behavior. Cache files contain image descriptions and are excluded from Git.

This fork has two main features:

1. When using a Gemma4 GGUF vision model to reverse-prompt images, you can choose whether to output think/reasoning content. By default, it is hidden.

2. When the input image is unchanged, all node parameters are unchanged, and the seed is fixed, the plugin skips reverse-prompt inference and reuses the previously generated prompt.

![Gemma4 thinking controls preview](./img/gemma4-thinking-controls.png)


## **1. Gemma4 Think Output Control**

This fork adds `启用思考` to `Llama-cpp Model Loader` and `输出think块` to `Llama-cpp Instruct`.

For normal Gemma4 image prompt reverse engineering, the recommended default is:

```text
启用思考: false
输出think块: false
```

This keeps the output closer to a normal prompt and avoids mixing `<think>...</think>` or channel reasoning text into the final prompt.

If you want to inspect the raw model output, you can enable `输出think块`.

This fork was mainly tested with:

[HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced](https://huggingface.co/HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced)

The model author's recommended sampling values are also used as defaults:

```text
temperature: 0.6
top_k: 64
top_p: 0.9
min_p: 0.05
repeat_penalty: 1.1
```


## **2. Fixed Seed Cache**

If `生成前控制` / `control_after_generate` is set to `fixed`, and the input image, model, prompt, and sampling parameters are unchanged, the plugin reuses the previous prompt instead of reading the image again.

This is useful while tuning a workflow: reverse-prompt once, then keep adjusting KSampler or other downstream nodes without waiting for repeated image inference.

If you change the image or any setting that affects the prompt result, the plugin automatically runs fresh inference.


## **3. Installation**

Clone this fork into ComfyUI `custom_nodes`:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/aiai-r/ComfyUI-llama-cpp_vlm.git
```

If your ComfyUI environment already runs the original plugin correctly, avoid reinstalling dependencies casually.

Only install requirements if your current environment is missing dependencies:

```bash
python -m pip install -r ComfyUI-llama-cpp_vlm/requirements.txt
```

After installing or updating, if old workflows do not show the new options, delete the old related nodes and recreate the following three nodes:

```text
Llama-cpp Model Loader
Llama-cpp Parameters
Llama-cpp Instruct
```


## **4. Credits**

- Original fork: [lihaoyun6/ComfyUI-llama-cpp_vlm](https://github.com/lihaoyun6/ComfyUI-llama-cpp_vlm)
- Upstream project: [kijai/ComfyUI-llama-cpp](https://github.com/kijai/ComfyUI-llama-cpp)
- [llama-cpp-python](https://github.com/JamePeng/llama-cpp-python)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
