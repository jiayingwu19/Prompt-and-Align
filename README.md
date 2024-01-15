# Data and Code for "Prompt-and-Align: Prompt-Based Social Alignment for Few-Shot Fake News Detection" (CIKM 2023)

This repo contains the data and code for the following paper:

Jiaying Wu, Shen Li, Ailin Deng, Miao Xiong, Bryan Hooi. Prompt-and-Align: Prompt-Based Social Alignment for Few-Shot Fake News Detection, ACM Conference on Information and Knowledge Management (CIKM) 2023. [![arXiv](https://img.shields.io/badge/arXiv-2309.16424-b31b1b.svg)](https://arxiv.org/abs/2309.16424)

## Abstract

Despite considerable advances in automated fake news detection, due to the timely nature of news, it remains a critical open question how to effectively predict the veracity of news articles based on limited fact-checks. Existing approaches typically follow a “Train-from-Scratch” paradigm, which is fundamentally bounded by the availability of large-scale annotated data. While expressive pre-trained language models (PLMs) have been adapted in a “Pre-Train-and-Fine-Tune” manner, the inconsistency between pre-training and downstream objectives also requires costly task-specific supervision. In this paper, we propose “Prompt-and-Align” (P&A), a novel prompt-based paradigm for few-shot fake news detection that jointly leverages the pre-trained knowledge in PLMs and the social context topology. Our approach mitigates label scarcity by wrapping the news article in a task-related textual prompt, which is then processed by the PLM to directly elicit task-specific knowledge. To supplement the PLM with social context without inducing additional training overheads, motivated by empirical observation on user veracity consistency (i.e., social users tend to consume news of the same veracity type), we further construct a news proximity graph among news articles to capture the veracity-consistent signals in shared readerships, and align the prompting predictions along the graph edges in a confidence-informed manner. Extensive experiments on three real-world benchmarks demonstrate that P&A sets new states-of-the-art for few-shot fake news detection performance by significant margins. 


## Requirements
We implement our proposed ''Prompt-and-Align'' (P&A) model and its variants based on PyTorch 1.8.0 with CUDA 11.1, and train them on a server running Ubuntu 18.04 with NVIDIA RTX 3090 GPU and Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz.

Main dependencies include:

```
python==3.7.0
numpy==1.21.4
torch==1.8.0+cu111
transformers==4.13.0
```

## Data

Our work is based on three benchmark datasets, including the `PolitiFact` and `GossipCop` datasets from the [FakeNewsNet benchmark](https://github.com/KaiDMML/FakeNewsNet), and the `FANG` dataset [(Nguyen et al., 2020)](https://github.com/nguyenvanhoang7398/FANG).

Extract the files in `data.tar.gz` to obtain an unzipped `data/` folder. The resultant `data/` should contain a list of .csv files and the following three folders: `adjs/`, `news_article_raw/` and `social_context_raw/`. 

**News Articles**

The .csv files under `data/` contain the news articles and their ground-truth labels. For each news article, label `0` represents real news, whereas label `1` represents fake news.

For each dataset, `data/[dataset_name]_train_[train_size].csv` is the training set given `train_size` annotated samples, and `data/[dataset_name]_test.csv` is the test set.

**Social Context**

The .pkl files under `data/user_news_graph/` each contain the following list: [train_conf, adj]. `train_conf` of shape `[train_size, 2]` refers to the ground-truth confidence matrix of the training samples (i.e., each row being `[1., 0.]` or `[0., 1.]`), and `adj_matrix` refers to the adjacency matrix of the news proximity graph.

The adjacency matrix of the news proximity graph is constructed via the following steps:

1. Collect the user-news repost records for all samples (including `train_size` training samples and the test samples), specifically in the form of `[sid, tid, uid]` for each line, which means that user `uid` has reposted news articles `sid` and the repost has Tweet ID of `tid`.
2. Filter the records with threshold $t_u$ to focus on the set of active social users. In our method, $t_u$ is set to 5, i.e., we only consider the social users with at least 5 repost records.
3. Construct a user engagement matrix $\mathbf{B}$ of size `[num_active_users, num_news]` where the value of element [k, m] in the matrix represents the number of reposts between active user $u_k$ and news article $T_m$.
4. Formulate the adjacency matrix as $\mathbf{A}_ {n}=\mathbf{B}^\top \mathbf{B}$, and conduct normalization to derive the final normalized adjacency matrix as $\mathbf{A}_ {\mathcal{T}} = \mathbf{D}_ {n}^{-\frac{1}{2}} \mathbf{A}_ {n} \mathbf{D}_ {n}^{-\frac{1}{2}}$.


**Raw Data**

We provide the raw user-news engagement records used to construct the above-mentioned news proximity graph from scratch, at `data/social_context_raw/[dataset_name]_socialcontext_train[train_size].csv`. There, each line is given as: `[sid,label,tid,uid]`, meaning that user `uid` has reposted news articles `sid` of veracity label (0: real; 1: fake), and the repost has Tweet ID of `tid`.

We also provide the raw news articles and metadata in the format of `[news_id,label,title,content]` at `data/news_articles_raw/[dataset_name]_full_train[train_size].csv`. Here, `news_id` refer to the article's ID in the FakeNewsNet / FANG dataset. In each file, the [train_size] training samples are placed first, followed by the test samples.

**Constructing the News Proximity Graph from the Raw Social Context** 

As an alternative to using our pre-processed adjacency matrices under `data/adjs/`, we provide a pre-processing script at `Process/adj_matrix_fewshot.py` to construct the matrices from scratch.  

Construct the adjacency matrices with the following command:

```bash
mkdir data/adjs_from_scratch
python Process/adj_matrix_fewshot.py
```

**Obtaining Raw Social Context via FakeNewsNet Scripts** 

The datasets used in our work are in line with the FakeNewsNet format. To retrieve the social engagements and auxiliary features related to social context, please follow the instructions and scripts given in the [FakeNewsNet GitHub repo](https://github.com/KaiDMML/FakeNewsNet).


## Run Prompt-and-Align

The P&A source code is provided in `prompt_and_align.py`. 

Start training with the following command:

```bash
sh run.sh
```


Results will be saved under the `logs/` directory. We also provide our original experiment logs under `logs/logs_archive/`.

## Contact

jiayingwu [at] u.nus.edu

## Citation

If you find this repo or our work useful for your research, please consider citing our paper

```
@inproceedings{wu2023prompt,
  author = {Wu, Jiaying and Li, Shen and Deng, Ailin and Xiong, Miao and Hooi, Bryan},
  title = {Prompt-and-Align: Prompt-Based Social Alignment for Few-Shot Fake News Detection},
  year = {2023},
  booktitle = {Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
  pages = {2726–2736}
}

```