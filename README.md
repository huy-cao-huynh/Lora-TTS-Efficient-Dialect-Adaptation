# LoRA-TTS: Efficient Dialect Adaptation

LoRA-TTS is a parameter efficient Text-to-Speech model fine-tuned using Low-Rank Adaptation. We build on a pre-trained text-to-speech model, and fine-tune on a small dataset of native Taiwanese speakers. We reduced the number of trainable parameters by 98% while maintaining better/similar results compared to full fine-tuning. This project demonstrates the efficiency and quality of LoRA fine-tuning.

# 📁 Contents

`datasets`
The datasets used to fine-tune our models. They're from the Mozilla Open Collective zh-TW audio dataset.

`references`
Contains the reference speaker file we used to generate audio from.

`slides`
Contains the original Figma .deck file as well as the corresponding .pdf file.

`testing`
Contains the code used to evaluate our models.

`training`
Contains the code used to train our models.

# 🧠 Specification
* PyTorch for training and evaluation
* PeFT: LoRA adapters for parameter efficient fine-tuning
* Dataset: Mozilla Data Collective
* Model: XTTS V2
