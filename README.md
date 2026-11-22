# OralMLLM-Bench

Code and materials for

> Wang R, Zhou S, Wang J, Xie W, Che X. **Cognitive evaluation of multimodal large language models in dental
> radiography.** *J Dent* 2026;107174. https://doi.org/10.1016/j.jdent.2026.107174

OralMLLM-Bench comprises 27 tasks on periapical, panoramic and lateral cephalometric radiographs, organized into
four cognitive task categories (perception, comprehension, prediction and decision-making). Six multimodal large
language models (MLLMs) were evaluated in a zero-shot setting. Model and clinician responses were scored manually
by two licensed dentists blinded to response source.

## Contents

| File | Description |
|---|---|
| `prompts.json` | System-level instruction and task-specific prompts |
| `run_inference.py` | Inference script |

## Inference

`run_inference.py` sends the system-level instruction, the image and the task-specific prompt of each image–task
query to the models and saves the responses.

```bash
python run_inference.py --manifest manifest.csv --images-root images/ --out responses.jsonl
```

`manifest.csv` lists the queries (`task_id, radiograph_id, image_path`). API keys are read from environment
variables. Running LLaVA-1.6-7B locally requires the packages in `requirements.txt`.

## Scoring and analysis

Scoring and statistical analysis followed the procedures described in the article and its supplementary material.

## Radiographs

Radiographs were obtained from six publicly available datasets. Access is subject to the terms and conditions of
each dataset provider.

| Modality | Dataset |
|---|---|
| Periapical | https://www.kaggle.com/datasets/muhammadsajad/periapical-xrays |
| Periapical | https://www.kaggle.com/datasets/nadaaglan/dental-periapical-x-rays |
| Panoramic | DENTEX: https://dentex.grand-challenge.org/data/ |
| Panoramic | OdontoAI: https://github.com/IvisionLab/OdontoAI-Open-Panoramic-Radiographs |
| Lateral cephalometric | Aariz: https://github.com/manwaarkhd/aariz |
| Lateral cephalometric | CL-Detection 2023: https://cl-detection2023.grand-challenge.org/ |

## License

Code: MIT License (see `LICENSE`).

## Citation

If you use this repository, please cite the article above.
