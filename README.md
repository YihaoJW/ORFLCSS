# Repo for ICASSP 2024 Paper

This repository contains the key implementation for our **ICASSP 2024** paper on improving **children’s Oral Reading Fluency (ORF)** assessment via **sub-sequence matching of acoustic word embeddings**.

**Paper:** *Improving Oral Reading Fluency Assessment Through Sub-Sequence Matching of Acoustic Word Embeddings*  [oai_citation:0‡DBLP](https://dblp.org/pid/154/1923?utm_source=chatgpt.com)  
**Authors:** Yihao Wang, Zhongdi Wu, Joseph Nese, Akihito Kamata, Vedant Nilabh, Eric C. Larson  [oai_citation:1‡DBLP](https://dblp.org/pid/154/1923?utm_source=chatgpt.com)  
**ICASSP 2024 pages:** 10766–10770  [oai_citation:2‡DBLP](https://dblp.org/pid/154/1923?utm_source=chatgpt.com)  
**PDF:** https://s2.smu.edu/~eclarson/pubs/2024_icassp_orf.pdf  [oai_citation:3‡SMU](https://s2.smu.edu/~eclarson/pubs/2024_icassp_orf.pdf?utm_source=chatgpt.com)

## Overview

We train an acoustic embedding model using unlabeled student oral reading audio, then apply **sub-sequence matching** in the embedding space to estimate **Words Correct Per Minute (WCPM)**—a core ORF metric—improving agreement with human scoring over ASR-only baselines.  [oai_citation:4‡SMU](https://s2.smu.edu/~eclarson/pubs/2024_icassp_orf.pdf?utm_source=chatgpt.com)

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{wang2024orf_subsequence_matching,
  title={Improving Oral Reading Fluency Assessment Through Sub-Sequence Matching of Acoustic Word Embeddings},
  author={Wang, Yihao and Wu, Zhongdi and Nese, Joseph and Kamata, Akihito and Nilabh, Vedant and Larson, Eric C.},
  booktitle={Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2024}
}
