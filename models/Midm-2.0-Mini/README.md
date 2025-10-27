---
license: mit
language:
- en
- ko
tags:
- KT
- K-intelligence
- Mi:dm
pipeline_tag: text-generation
library_name: transformers
---

<p align="center">
<br>
    <span style="font-size: 60px; font-weight: bold;">Mi:dm 2.0 Mini</span>
</br>
</p>
<p align="center">
🤗 <a href="https://huggingface.co/collections/K-intelligence/mi-dm-20-6866406c301e5f45a6926af8">Mi:dm 2.0 Models</a> |
📜 <a href="https://github.com/K-intelligence-Midm/Midm-2.0/blob/main/Mi_dm2_0__technical_report.pdf">Mi:dm 2.0 Technical Report</a> |
📕 <a href="https://kode.kt.com/blog/article/3935">Mi:dm 2.0 Technical Blog</a>
</p>

<p align="center"><sub>*To be released soon</sub></p>

<br>

# News 📢

- 🔜 _(Coming Soon!) GGUF format model files will be available soon for easier local deployment._
- 📕`2025/08/08`: Published a technical blog article about Mi:dm 2.0 Model.
- ⚡️`2025/07/04`: Released Mi:dm 2.0 Model collection on Hugging Face🤗.
<br>
<br>
# Table of Contents

- ___Overview___
    - [Mi:dm 2.0](#midm-20)
    - [Quickstart](#quickstart)
    - [Evaluation](#evaluation)
- ___Usage___
    - [Run on Friendly.AI](#run-on-friendliai)
    - [Run on Your Local Machine](#run-on-your-local-machine)
    - [Deployment](#deployment)
    - [Tutorials](#tutorials)
- ___More Information___
    - [Limitation](#limitation)
    - [License](#license)
    - [Contact](#contact)

<br>
<br>

# Overview

## Mi:dm 2.0

**Mi:dm 2.0** is a __"Korea-centric AI"__ model developed using KT's proprietary technology. The term __"Korea-centric AI"__ refers to a model that deeply internalizes the unique values, cognitive frameworks, and commonsense reasoning inherent to Korean society. It goes beyond simply processing or generating Korean text—it reflects a deeper understanding of the socio-cultural norms and values that define Korean society.

Mi:dm 2.0 is released in two versions:

- **Mi:dm 2.0 Base**  
  An 11.5B parameter dense model designed to balance model size and performance.  
  It extends an 8B-scale model by applying the Depth-up Scaling (DuS) method, making it suitable for real-world applications that require both performance and versatility.

- **Mi:dm 2.0 Mini**  
  A lightweight 2.3B parameter dense model optimized for on-device environments and systems with limited GPU resources.  
  It was derived from the Base model through pruning and distillation to enable compact deployment.


> [!Note]
> Neither the pre-training nor the post-training data includes KT users' data.


<br>

## Quickstart

Here is the code snippet to run conversational inference with the model:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

model_name = "K-intelligence/Midm-2.0-Mini-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
generation_config = GenerationConfig.from_pretrained(model_name)

prompt = "KT에 대해 소개해줘"

# message for inference
messages = [
    {"role": "system", 
     "content": "Mi:dm(믿:음)은 KT에서 개발한 AI 기반 어시스턴트이다."},
    {"role": "user", "content": prompt}
]

input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
)

output = model.generate(
    input_ids.to("cuda"),
    generation_config=generation_config,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=128,
    do_sample=False,
)
print(tokenizer.decode(output[0]))
```

> [!NOTE]
> The `transformers` library should be version `4.45.0` or higher.

<br>
<br>

## Evaluation


### Korean

<!-- first half table-->
<table>
<tr>
  <th rowspan="2">Model</th>
  <th colspan="5" align="center">Society & Culture</th>
  <th colspan="3" align="center">General Knowledge</th>
  <th colspan="3" align="center">Instruction Following</th>
</tr>
<tr>
  <th align="center">K-Refer<sup>*</sup></th>
  <th align="center">K-Refer-Hard<sup>*</sup></th>
  <th align="center">Ko-Sovereign<sup>*</sup></th>
  <th align="center">HAERAE</th>
  <th align="center">Avg.</th>
  <th align="center">KMMLU</th>
  <th align="center">Ko-Sovereign<sup>*</sup></th>
  <th align="center">Avg.</th>
  <th align="center">Ko-IFEval</th>
  <th align="center">Ko-MTBench</th>
  <th align="center">Avg.</th>
</tr>

<!-- Small Models -->
<tr>
  <td><strong>Qwen3-4B</strong></td>
  <td align="center">53.6</td>
  <td align="center">42.9</td>
  <td align="center">35.8</td>
  <td align="center">50.6</td>
  <td align="center">45.7</td>
  <td align="center"><strong>50.6</strong></td>
  <td align="center"><strong>42.5</strong></td>
  <td align="center"><strong>46.5</strong></td>
  <td align="center"><strong>75.9</strong></td>
  <td align="center">63.0</td>
  <td align="center">69.4</td>
</tr>
<tr>
  <td><strong>Exaone-3.5-2.4B-inst</strong></td>
  <td align="center">64.0</td>
  <td align="center"><strong>67.1</strong></td>
  <td align="center"><strong>44.4</strong></td>
  <td align="center">61.3</td>
  <td align="center"><strong>59.2</strong></td>
  <td align="center">43.5</td>
  <td align="center">42.4</td>
  <td align="center">43.0</td>
  <td align="center">65.4</td>
  <td align="center"><strong>74.0</strong></td>
  <td align="center">68.9</td>
</tr>
<tr>
  <td><strong>Mi:dm 2.0-Mini-inst</strong></td>
  <td align="center"><strong>66.4</strong></td>
  <td align="center">61.4</td>
  <td align="center">36.7</td>
  <td align="center"><strong>70.8</strong></td>
  <td align="center">58.8</td>
  <td align="center">45.1</td>
  <td align="center">42.4</td>
  <td align="center">43.8</td>
  <td align="center">73.3</td>
  <td align="center"><strong>74.0</strong></td>
  <td align="center"><strong>73.6</strong></td>
</tr>

<!-- Spacer row -->
<tr><td colspan="13"> </td></tr>

<!-- Large Models -->
<tr>
  <td><strong>Qwen3-14B</strong></td>
  <td align="center">72.4</td>
  <td align="center">65.7</td>
  <td align="center">49.8</td>
  <td align="center">68.4</td>
  <td align="center">64.1</td>
  <td align="center">55.4</td>
  <td align="center">54.7</td>
  <td align="center">55.1</td>
  <td align="center"><strong>83.6</strong></td>
  <td align="center">71</td>
  <td align="center">77.3</td>
</tr>
<tr>
  <td><strong>Llama-3.1-8B-inst</strong></td>
  <td align="center">43.2</td>
  <td align="center">36.4</td>
  <td align="center">33.8</td>
  <td align="center">49.5</td>
  <td align="center">40.7</td>
  <td align="center">33.0</td>
  <td align="center">36.7</td>
  <td align="center">34.8</td>
  <td align="center">60.1</td>
  <td align="center">57</td>
  <td align="center">58.5</td>
</tr>
<tr>
  <td><strong>Exaone-3.5-7.8B-inst</strong></td>
  <td align="center">71.6</td>
  <td align="center">69.3</td>
  <td align="center">46.9</td>
  <td align="center">72.9</td>
  <td align="center">65.2</td>
  <td align="center">52.6</td>
  <td align="center">45.6</td>
  <td align="center">49.1</td>
  <td align="center">69.1</td>
  <td align="center">79.6</td>
  <td align="center">74.4</td>
</tr>
<tr>
  <td><strong>Mi:dm 2.0-Base-inst</strong></td>
  <td align="center"><strong>89.6</strong></td>
  <td align="center"><strong>86.4</strong></td>
  <td align="center"><strong>56.3</strong></td>
  <td align="center"><strong>81.5</strong></td>
  <td align="center"><strong>78.4</strong></td>
  <td align="center"><strong>57.3</strong></td>
  <td align="center"><strong>58.0</strong></td>
  <td align="center"><strong>57.7</strong></td>
  <td align="center">82</td>
  <td align="center"><strong>89.7</strong></td>
  <td align="center"><strong>85.9</strong></td>
</tr>
</table>

<!-- second half table-->
<table>
<tr>
  <th rowspan="2" align="center">Model</th>
  <th colspan="5" align="center">Comprehension</th>
  <th colspan="5" align="center">Reasoning</th>
</tr>
<tr>
  <th align="center">K-Prag<sup>*</sup></th>
  <th align="center">K-Refer-Hard<sup>*</sup></th>
  <th align="center">Ko-Best</th>
  <th align="center">Ko-Sovereign<sup>*</sup></th>
  <th align="center">Avg.</th>
  <th align="center">Ko-Winogrande</th>
  <th align="center">Ko-Best</th>
  <th align="center">LogicKor</th>
  <th align="center">HRM8K</th>
  <th align="center">Avg.</th>
</tr>

<!-- Small Models -->
<tr>
  <td><strong>Qwen3-4B</strong></td>
  <td align="center"><strong>73.9<strong></td>
  <td align="center">56.7</td>
  <td align="center"><strong>91.5</strong></td>
  <td align="center"><strong>43.5</strong></td>
  <td align="center"><strong>66.6</strong></td>
  <td align="center"><strong>67.5</strong></td>
  <td align="center"><strong>69.2</strong></td>
  <td align="center">5.6</td>
  <td align="center"><strong>56.7</strong></td>
  <td align="center"><strong>43.8</strong></td>
</tr>
<tr>
  <td><strong>Exaone-3.5-2.4B-inst</strong></td>
  <td align="center">68.7</td>
  <td align="center"><strong>58.5</strong></td>
  <td align="center">87.2</td>
  <td align="center">38.0</td>
  <td align="center">62.5</td>
  <td align="center">60.3</td>
  <td align="center">64.1</td>
  <td align="center">7.4</td>
  <td align="center">38.5</td>
  <td align="center">36.7</td>
</tr>
<tr>
  <td><strong>Mi:dm 2.0-Mini-inst</strong></td>
  <td align="center">69.5</td>
  <td align="center">55.4</td>
  <td align="center">80.5</td>
  <td align="center">42.5</td>
  <td align="center">61.9</td>
  <td align="center">61.7</td>
  <td align="center">64.5</td>
  <td align="center"><strong>7.7</strong></td>
  <td align="center">39.9</td>
  <td align="center">37.4</td>
</tr>

<!-- Visual Spacer -->
<tr><td colspan="11"> </td></tr>

<!-- Large Models -->
<tr>
  <td><strong>Qwen3-14B</strong></td>
  <td align="center"><strong>86.7</strong></td>
  <td align="center"><strong>74.0</strong></td>
  <td align="center">93.9</td>
  <td align="center">52.0</td>
  <td align="center"><strong>76.8</strong></td>
  <td align="center"><strong>77.2</strong></td>
  <td align="center"><strong>75.4</strong></td>
  <td align="center">6.4</td>
  <td align="center"><strong>64.5</strong></td>
  <td align="center"><strong>48.8</strong></td>
</tr>
<tr>
  <td><strong>Llama-3.1-8B-inst</strong></td>
  <td align="center">59.9</td>
  <td align="center">48.6</td>
  <td align="center">77.4</td>
  <td align="center">31.5</td>
  <td align="center">51.5</td>
  <td align="center">40.1</td>
  <td align="center">26.0</td>
  <td align="center">2.4</td>
  <td align="center">30.9</td>
  <td align="center">19.8</td>
</tr>
<tr>
  <td><strong>Exaone-3.5-7.8B-inst</strong></td>
  <td align="center">73.5</td>
  <td align="center">61.9</td>
  <td align="center">92.0</td>
  <td align="center">44.0</td>
  <td align="center">67.2</td>
  <td align="center">64.6</td>
  <td align="center">60.3</td>
  <td align="center"><strong>8.6</strong></td>
  <td align="center">49.7</td>
  <td align="center">39.5</td>
</tr>
<tr>
  <td><strong>Mi:dm 2.0-Base-inst</strong></td>
  <td align="center">86.5</td>
  <td align="center">70.8</td>
  <td align="center"><strong>95.2</strong></td>
  <td align="center"><strong>53.0</strong></td>
  <td align="center">76.1</td>
  <td align="center">75.1</td>
  <td align="center">73.0</td>
  <td align="center"><strong>8.6</strong></td>
  <td align="center">52.9</td>
  <td align="center">44.8</td>
</tr>
</table>

`*` indicates KT proprietary evaluation resources.

<br>


### English


<table>
<tr>
  <th rowspan="2" align="center">Model</th>
  <th align="center">Instruction</th>
  <th colspan="4" align="center">Reasoning</th>
  <th align="center">Math</th>
  <th align="center">Coding</th>
  <th colspan="3" align="center">General Knowledge</th>
</tr>
<tr>
  <th align="center">IFEval</th>
  <th align="center">BBH</th>
  <th align="center">GPQA</th>
  <th align="center">MuSR</th>
  <th align="center">Avg.</th>
  <th align="center">GSM8K</th>
  <th align="center">MBPP+</th>
  <th align="center">MMLU-pro</th>
  <th align="center">MMLU</th>
  <th align="center">Avg.</th>
</tr>

<!-- Small Models -->
<tr>
  <td><strong>Qwen3-4B</strong></td>
  <td align="center">79.7</td>
  <td align="center"><strong>79.0</strong></td>
  <td align="center"><strong>39.8</strong></td>
  <td align="center"><strong>58.5</strong></td>
  <td align="center"><strong>59.1</strong></td>
  <td align="center"><strong>90.4</strong></td>
  <td align="center">62.4</td>
  <td align="center">-</td>
  <td align="center"><strong>73.3</strong></td>
  <td align="center"><strong>73.3</strong></td>
</tr>
<tr>
  <td><strong>Exaone-3.5-2.4B-inst</strong></td>
  <td align="center"><strong>81.1</strong></td>
  <td align="center">46.4</td>
  <td align="center">28.1</td>
  <td align="center">49.7</td>
  <td align="center">41.4</td>
  <td align="center">82.5</td>
  <td align="center">59.8</td>
  <td align="center">-</td>
  <td align="center">59.5</td>
  <td align="center">59.5</td>
</tr>
<tr>
  <td><strong>Mi:dm 2.0-Mini-inst</strong></td>
  <td align="center">73.6</td>
  <td align="center">44.5</td>
  <td align="center">26.6</td>
  <td align="center">51.7</td>
  <td align="center">40.9</td>
  <td align="center">83.1</td>
  <td align="center"><strong>60.9</strong></td>
  <td align="center">-</td>
  <td align="center">56.5</td>
  <td align="center">56.5</td>
</tr>

<tr><td colspan="11">&nbsp;</td></tr>

<!-- Large Models -->
<tr>
  <td><strong>Qwen3-14B</strong></td>
  <td align="center">83.9</td>
  <td align="center"><strong>83.4</strong></td>
  <td align="center"><strong>49.8</strong></td>
  <td align="center"><strong>57.7</strong></td>
  <td align="center"><strong>63.6</strong></td>
  <td align="center">88.0</td>
  <td align="center">73.4</td>
  <td align="center"><strong>70.5</strong></td>
  <td align="center"><strong>82.7</strong></td>
  <td align="center"><strong>76.6</strong></td>
</tr>
<tr>
  <td><strong>Llama-3.1-8B-inst</strong></td>
  <td align="center">79.9</td>
  <td align="center">60.3</td>
  <td align="center">21.6</td>
  <td align="center">50.3</td>
  <td align="center">44.1</td>
  <td align="center">81.2</td>
  <td align="center"><strong>81.8</strong></td>
  <td align="center">47.6</td>
  <td align="center">70.7</td>
  <td align="center">59.2</td>
</tr>
<tr>
  <td><strong>Exaone-3.5-7.8B-inst</strong></td>
  <td align="center">83.6</td>
  <td align="center">50.1</td>
  <td align="center">33.1</td>
  <td align="center">51.2</td>
  <td align="center">44.8</td>
  <td align="center">81.1</td>
  <td align="center">79.4</td>
  <td align="center">40.7</td>
  <td align="center">69.0</td>
  <td align="center">54.8</td>
</tr>
<tr>
  <td><strong>Mi:dm 2.0-Base-inst</strong></td>
  <td align="center"><strong>84.0</strong></td>
  <td align="center">77.7</td>
  <td align="center">33.5</td>
  <td align="center">51.9</td>
  <td align="center">54.4</td>
  <td align="center"><strong>91.6</strong></td>
  <td align="center">77.5</td>
  <td align="center">53.3</td>
  <td align="center">73.7</td>
  <td align="center">63.5</td>
</tr>
</table>

<br>

# Usage

## Run on Friendli.AI
You can try our model immediately via `Friendli.AI`. Simply click `Deploy` and then `Friendli Endpoints`. 

> [!Note]
> Please note that a login to `Friendli.AI` is required after your fifth chat interaction.

<p>
  <img src="./assets/image_1.png" alt="Left Image" width="36%" style="display:inline-block; margin-right:2%">
  <img src="./assets/image_2.png" alt="Right Image" width="36%" style="display:inline-block">
</p>

## Run on Your Local Machine
We provide a detailed description about running Mi:dm 2.0 on your local machine using llama.cpp, LM Studio, and Ollama. Please check our [github](https://github.com/K-intelligence-Midm/Midm-2.0) for more information


## Deployment

To serve Mi:dm 2.0 using [vLLM](https://github.com/vllm-project/vllm)(`>=0.8.0`) with an OpenAI-compatible API:
```bash
vllm serve K-intelligence/Midm-2.0-Mini-Instruct
```


## Tutorials
To help our end-users easily use Mi:dm 2.0, we have provided comprehensive tutorials on [github](https://github.com/K-intelligence-Midm/Midm-2.0). 
<br>

<br>
<br>

# More Information

## Limitation
* The training data for both Mi:dm 2.0 models consists primarily of English and Korean. Understanding and generation in other languages are not guaranteed.
  
* The model is not guaranteed to provide reliable advice in fields that require professional expertise, such as law, medicine, or finance.

* Researchers have made efforts to exclude unethical content from the training data — such as profanity, slurs, bias, and discriminatory language. However, despite these efforts, the model may still produce inappropriate expressions or factual inaccuracies.


## License

Mi:dm 2.0 is licensed under the [MIT License](./LICENSE).
 
<!-- ### Citation
 
```
@misc{,
      title={}, 
      author={},
      year={2025},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={}, 
}
``` -->
## Contact 
Mi:dm 2.0 Technical Inquiries: midm-llm@kt.com

<br>