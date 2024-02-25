### Human annotated highlights and corrections

1. Dev Set
```angular2html
annotation/annotation_dev.json
```

2. Test Set
```angular2html
annotation/annotation_test.json
```

Data fields:
```angular2html
"numbered_instruction": tokenized instruction with token index number
"id2label": 
key: span indices, e.g. "5-6" means the 5th and 6th tokens in the instruction
value: 1 (correct), 0 (intrinsic hallucination), -1 (extrinsic hallucination)
"id2gold_alternative":
key: span indices, e.g. "5-6" means the 5th and 6th tokens in the instruction
value: the corrected phrase consistent with visual route
```


### Human evaluation dataset on navigation 

Annotations on the Test Set: 
```angular2html
annotation/5models_human_eval_final.json
```

It contains results of 5 evaluated models:

1) No communication
2) HEAR (no suggestion)
3) HEAR
4) Oracle (no suggestion)
5) Oracle

Data fields:
```angular2html
key: model name
value: human evaluation results for this model
"human_annotations": list of 5 human annotations for each instruction. "path" is human followed path
"human_scores": list of 5 human scores computed from each annotation for each instruction
```


