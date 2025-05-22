# GazeVLM

## Installation

```bash
conda create --name gazevlm python=3.12
conda activate gazevlm
pip install -e transformers
conda install pytorch pillow accelerator
```

## Inference on Public Dataset

We have implemented our framework on two real-world eye gaze-based datasets: [AiR-D](https://github.com/szzexpoi/AiR) (images and QAs from [GQA](https://cs.stanford.edu/people/dorarad/gqa/download.html)) and [VQA-MHUG](https://www.collaborative-ai.org/research/datasets/VQA-MHUG/) (images and QAs from [VQAv2](https://visualqa.org/download.html) val2017).

### AiR-D Dataset

[Gaze points in csv](assets/AiR-D/fixation_csv/SAMPLE-question_fixation_mean_correct.csv) consists of three attributes:

| Question ID | X           | Y           |
|-------------|-------------|-------------|
| 00126851    | 266.281712  | 281.020663  |
| 00137872    | 166.859517  | 174.318234  |
| 00137901    | 248.836341  | 258.729393  |
| 00148535    | 185.683648  | 201.719850  |
| 00155957    | 339.662321  | 200.378664  |

The question and answer set for the corresponding question ID can be found in [QA json file](assets/AiR-D/GQA/questions/SAMPLE-val_balanced_questions.json), and the images can be found in [image directory](assets/AiR-D/GQA/images):

`{"**00126851**": {"semantic": [{"operation": "select", "dependencies": [], "argument": "pond (1039893)"}, {"operation": "relate", "dependencies": [0], "argument": "cart,near,s (1039872)"}, {"operation": "exist", "dependencies": [1], "argument": "?"}], "entailed": [], "equivalent": ["00126851"], **"question": "Are there carts near the pond?"**, **"imageId": "1159529"**, "isBalanced": true, "groups": {"global": null, "local": "13-pond_cart"}, **"answer": "yes"**, "semanticStr": "select: pond (1039893)->relate: cart,near,s (1039872) [0]->exist: ? [1]", "annotations": {"answer": {}, "question": {"2": "1039872", "5": "1039893"}, "fullAnswer": {"4": "1039872", "7": "1039893"}}, "types": {"detailed": "existRelS", "semantic": "rel", "structural": "verify"}, "fullAnswer": "Yes, there is a cart near the pond."}, "00137872": ...}`

Run the inference for AiR-D:

```Shell
python whole-inference-aird.py
```

### VQA-MHUG Datasets

[Gaze points](assets/VQA-MHUG/fixation_csv/SAMPLE-question-fixation-vqa-mhug-mean-correct.csv), [QA json file](assets/VQA-MHUG/VQA-v2/questions/SAMPLE-v2_mscoco_val2014_refined_qa.json), and the [images](assets/VQA-MHUG/VQA-v2/images) are similar to AiR-D dataset.

Run the inference for VQA-MHUG:

```Shell
python whole-inference-mhug.py
```

## Sample Inference

![Image](assets/2358679.jpg)

- Question ID: 03627620
- Question: What is that bicycle leaning against?
- Image: 2358679.jpg
- Answer: pole
- Full answer: The bicycle is leaning against the pole.


### Compute or extract the gaze point (e.g., average all gaze points)

![Image](assets/gaze-points.png)

### Run GazeVLM on single QA set

```Shell
python single-inference-sample.py
```

> The bicycle is leaning against a green trash can.

## Acknowledgement

- [Transformers](https://github.com/huggingface/transformers): the codebase we built upon
- [LLaVA](https://github.com/haotian-liu/LLaVA.git): the model architecture we borrowed from
- [Model Checkpoints](https://huggingface.co/llava-hf): the pretrained models we used
- [AiR-D Dataset](https://github.com/szzexpoi/AiR)
- [VQA-MHUG Dataset](https://www.collaborative-ai.org/research/datasets/VQA-MHUG/)
