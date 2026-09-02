# Clean Caption Evaluation

The clean evaluator uses `PTBTokenizer` followed by the `pycocoevalcap`
BLEU, METEOR, ROUGE-L, and CIDEr implementations. It requires exact equality
between prediction and reference video IDs and evaluates against every
available reference caption. No external caption rewriting is part of this
evaluation path.

## Reproduce

From a YOLO-World checkout containing the separately transferred dataset:

```bash
python scripts/yoloworld_lightning/scripts/evaluate_captions.py \
  --dataset MSR-VTT \
  --references scripts/yoloworld_lightning/data/msrvtt/anno/test.json \
  --predictions scripts/yoloworld_lightning/data/msrvtt/txt \
  --output scripts/yoloworld_lightning/results/msrvtt_clean_eval.json
```

## Verified Result

The archived MSR-VTT predictions contain one direct model-generated caption
for each of the 2,990 test videos. The test path that produced these files had
external caption polishing disabled.

| Dataset | B-1 | B-2 | B-3 | B-4 | METEOR | ROUGE-L | CIDEr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MSR-VTT | 90.90 | 71.36 | 52.86 | 37.42 | 27.37 | 63.85 | 53.13 |

The full-precision result and reference annotation hash are stored in
`scripts/yoloworld_lightning/results/msrvtt_clean_eval.json`.

MSVD is deliberately not reported yet. Its final prediction file and final
checkpoint are not present in the workspace, so a new number cannot currently
be reproduced. The previously stated METEOR range of 38--42 was an expectation,
not a measured result.
