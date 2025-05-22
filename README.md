# GazeVLM

## Installation

```bash
conda create --name gazevlm python=3.12
conda activate gazevlm
pip install -e transformers
pip install -e lmms-eval
```

## Inference on AiR-D Dataset
The dataset originates from [AiR-D](https://github.com/szzexpoi/AiR).
```Shell
python whole-inference-aird.py
```

## Inference on VQA-MHUG Dataset
The dataset originates from [VQA-MHUG](https://www.collaborative-ai.org/research/datasets/VQA-MHUG/).
```Shell
python whole-inference-mhug.py
```

## Single Inference

![Image](https://github.com/user-attachments/assets/7d035e83-be6e-4331-b325-3f5ec396415d)

- Question ID: 03627620
- Question: What is that bicycle leaning against?
- Image: 2358679.jpg
- Answer: pole
- Full answer: The bicycle is leaning against the pole.


### Compute or extract the gaze point (e.g., average all gaze points)

![Image](https://github.com/user-attachments/assets/abcf6080-609a-409f-b236-34d930183e21)

### Run GazeVLM on single QA set
```Shell
python single-inference-sample.py
```
> The bicycle is leaning against a green trash can.

