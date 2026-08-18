---
dataset_info:
- config_name: all
  features:
  - name: prompt
    dtype: string
  - name: solution
    dtype: string
  - name: data_source
    dtype: string
  - name: source_prompt
    list:
    - name: content
      dtype: string
    - name: role
      dtype: string
  - name: ability
    dtype: string
  - name: reward_model
    struct:
    - name: ground_truth
      dtype: string
    - name: style
      dtype: string
  - name: extra_info
    struct:
    - name: index
      dtype: string
  splits:
  - name: train
    num_bytes: 15775526
    num_examples: 17398
  download_size: 6271739
  dataset_size: 15775526
- config_name: cn
  features:
  - name: prompt
    dtype: string
  - name: solution
    dtype: string
  - name: data_source
    dtype: string
  - name: source_prompt
    list:
    - name: content
      dtype: string
    - name: role
      dtype: string
  - name: ability
    dtype: string
  - name: reward_model
    struct:
    - name: ground_truth
      dtype: string
    - name: style
      dtype: string
  - name: extra_info
    struct:
    - name: index
      dtype: string
  splits:
  - name: train
    num_bytes: 2975932.6550178183
    num_examples: 3282
  download_size: 1005025
  dataset_size: 2975932.6550178183
- config_name: en
  features:
  - name: prompt
    dtype: string
  - name: solution
    dtype: string
  - name: data_source
    dtype: string
  - name: source_prompt
    list:
    - name: content
      dtype: string
    - name: role
      dtype: string
  - name: ability
    dtype: string
  - name: reward_model
    struct:
    - name: ground_truth
      dtype: string
    - name: style
      dtype: string
  - name: extra_info
    struct:
    - name: index
      dtype: string
  splits:
  - name: train
    num_bytes: 12799593.344982183
    num_examples: 14116
  download_size: 5232060
  dataset_size: 12799593.344982183
configs:
- config_name: all
  data_files:
  - split: train
    path: all/train-*
  default: true
- config_name: cn
  data_files:
  - split: train
    path: cn/train-*
- config_name: en
  data_files:
  - split: train
    path: en/train-*
---

# Dataset Card for DAPO-Math-17k-Processed

This is a processed version of [BytedTsinghua-SIA/DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) where we have:

* Deduplicated the prompts
* Reformatted the prompts and ground truth answers to be compatible with TRL's GRPO trainer

We have also derived pure English and Chinese subsets.

The full dataset processing logic can be found in create_dataset.py.

If you find this dataset useful in your work, please cite the original source with:

```
@misc{yu2025dapoopensourcellmreinforcement,
      title={DAPO: An Open-Source LLM Reinforcement Learning System at Scale},
      author={Qiying Yu and Zheng Zhang and Ruofei Zhu and Yufeng Yuan and Xiaochen Zuo and Yu Yue and Tiantian Fan and Gaohong Liu and Lingjun Liu and Xin Liu and Haibin Lin and Zhiqi Lin and Bole Ma and Guangming Sheng and Yuxuan Tong and Chi Zhang and Mofan Zhang and Wang Zhang and Hang Zhu and Jinhua Zhu and Jiaze Chen and Jiangjie Chen and Chengyi Wang and Hongli Yu and Weinan Dai and Yuxuan Song and Xiangpeng Wei and Hao Zhou and Jingjing Liu and Wei-Ying Ma and Ya-Qin Zhang and Lin Yan and Mu Qiao and Yonghui Wu and Mingxuan Wang},
      year={2025},
      eprint={2503.14476},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.14476},
}
```