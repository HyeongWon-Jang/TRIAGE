# Third-party code

This repository redistributes and modifies code from several other projects. The root `LICENSE` (Apache 2.0) applies to the parts written by the authors; the parts listed below remain under the licences named here.

## verl (Apache License 2.0)

`triage_rl/verl_overlay/` contains copies of files from [verl](https://github.com/volcengine/verl), several of them modified. Each modified file carries a notice at the top. verl's attribution notice, reproduced here as Apache 2.0 section 4(d) requires:

```
Copyright 2023-2024 Bytedance Ltd. and/or its affiliates
```

## Hugging Face open-r1 and TRL (Apache License 2.0)

`triage_sft/{sft.py,configs.py,utils.py}` are modified from [open-r1](https://github.com/huggingface/open-r1), which itself builds on [TRL](https://github.com/huggingface/trl); the SFT trainer they drive is TRL's. All three retain the upstream copyright header and a statement that they were changed.

## Raindrop (MIT License)

`triage_sft/data/P12/process_script/ParseData.py` and `triage_sft/data/P19/process_script/Generate_splitID.py` derive from [Raindrop](https://github.com/mims-harvard/Raindrop).

```
MIT License

Copyright (c) 2021 Zitnik Lab @ Harvard

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
```

## Datasets

- P12: PhysioNet Challenge 2012, Open Data Commons Attribution License v1.0 ([licence](https://physionet.org/content/challenge-2012/view-license/1.0.0/)).
- P19: PhysioNet Challenge 2019, Creative Commons Attribution 4.0 International Public License ([licence](https://physionet.org/content/challenge-2019/view-license/1.0.0/)). Please cite Reyna et al., *Early Prediction of Sepsis From Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019*, Critical Care Medicine, 2019.
- MIMIC-III, MIMIC-IV and eICU-CRD: credentialed access via PhysioNet; nothing derived from them is redistributed here.
