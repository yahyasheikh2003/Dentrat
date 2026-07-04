| Metric                           | Value                                          |
|:---------------------------------|:-----------------------------------------------|
| Model                            | dental_model_finetuned.pth                     |
| Test folder                      | C:\Users\hp\dental-anomaly-detection\test\test |
| Confidence threshold             | 0.05                                           |
| Accuracy                         | 0.6567                                         |
| Balanced Accuracy                | 0.6613                                         |
| MCC                              | 0.9080                                         |
| Precision (macro/micro/weighted) | 0.5450 / 0.9455 / 0.9467                       |
| Recall (macro/micro/weighted)    | 0.4723 / 0.9455 / 0.9455                       |
| F1 (macro/micro/weighted)        | 0.4917 / 0.9455 / 0.9375                       |
| mAP @ IoU=0.50                   | 0.2569                                         |
| mAP @ IoU=0.50:0.95              | 0.1066                                         |
| Avg Inference (ms) / FPS         | 0.00 / 0.00                                    |
| Parameters / Model Size          | 41,329,911 / 158.17 MB                         |

| Class Name             |   Support |   Precision |   Recall |   F1 Score |   Specificity |    FPR |    FNR |   AP@0.50 |   Avg IoU (TP) |   ROC-AUC |   TP |   FP |   FN |
|:-----------------------|----------:|------------:|---------:|-----------:|--------------:|-------:|-------:|----------:|---------------:|----------:|-----:|-----:|-----:|
| Caries                 |       165 |      0.0160 |   0.1628 |     0.0292 |        0.4609 | 0.5391 | 0.8372 |    0.0435 |         0.6197 |       nan |   21 | 1289 |  108 |
| Impacted Teeth         |       444 |      0.2490 |   0.8460 |     0.3847 |        0.4258 | 0.5742 | 0.1540 |    0.8908 |         0.7514 |       nan |  368 | 1110 |   67 |
| Broken Down Crown/Root |       520 |      0.0433 |   0.8015 |     0.0821 |        0.0751 | 0.9249 | 0.1985 |    0.2625 |         0.6889 |       nan |  416 | 9195 |  103 |
| Infection              |        93 |      0.0000 |   0.0000 |     0.0000 |        1.0000 | 0.0000 | 1.0000 |    0.0000 |         0.0000 |       nan |    0 |    0 |   92 |
| Fractured Teeth        |         0 |      0.0000 |   0.0000 |     0.0000 |        1.0000 | 0.0000 | 0.0000 |  nan      |         0.0000 |       nan |    0 |    0 |    0 |
| Periodontal Bone Loss  |        45 |      0.0276 |   0.6136 |     0.0527 |        0.5618 | 0.4382 | 0.3864 |    0.0878 |         0.6830 |       nan |   27 |  953 |   17 |
| Other Abnormalities    |         0 |      0.0000 |   0.0000 |     0.0000 |        0.8768 | 0.1232 | 0.0000 |  nan      |         0.0000 |       nan |    0 |  178 |    0 |