# Final Study Report (20 Study Stations, Frozen QC v3, DATE-Block CIs)

Primary method (fixed in advance): `equal_neural_ens` (5-family equal average of `gru, lstm, bilstm, tcn, cnn_lstm`, 5 seeds `0..4`), `direct` target, `none` weather.

## 1. Run Summary, Persistence & All Baselines

```text
                               run_id  N_test  date_blks  persist_RMSE  same_hour_yesterday_skill  climatology_skill  ridge_skill  gbm_direct_skill  xgb_direct_skill  gbm_residual_skill  xgb_residual_skill best_tree_choice  best_tree_RMSE  best_tree_skill_RMSE  equal_neural_ens_RMSE  equal_neural_ens_skill_RMSE  stack_RMSE  stack_skill_RMSE
         P2_2023_ALL20_h1_direct_none  151279        365        16.836                        NaN            -288.62         7.31              7.22              7.68               11.30               11.18     gbm_residual          14.933                 11.30                 14.588                        13.35      14.562             13.51
    P2_2023_ALL20_h1_direct_openmeteo  151279        365        16.836                        NaN            -288.62         7.76              8.01              8.07               11.79               12.37     xgb_residual          14.752                 12.37                 14.452                        14.16      14.517             13.77
         P2_2023_ALL20_h6_direct_none  149831        365        42.552                       9.57             -53.74        26.18             30.15             30.22               30.35               30.10       xgb_direct          29.692                 30.22                 29.576                        30.49      29.369             30.98
    P2_2023_ALL20_h6_direct_openmeteo  149831        365        42.552                       9.57             -53.74        27.42             32.20             32.30               32.55               32.55       gbm_direct          28.849                 32.20                 28.165                        33.81      27.952             34.31
        P2_2023_ALL20_h12_direct_none  148395        365        48.621                      20.77             -34.64        30.40             32.74             33.08               32.46               32.39       xgb_direct          32.538                 33.08                 32.468                        33.22      32.341             33.48
   P2_2023_ALL20_h12_direct_openmeteo  148395        365        48.621                      20.77             -34.64        31.60             35.13             35.32               34.70               34.91       gbm_direct          31.543                 35.13                 30.835                        36.58      30.642             36.98
        P2_2023_ALL20_h24_direct_none  146751        365        38.837                       0.00             -68.43         9.90             12.25             12.64               11.93               11.89       xgb_direct          33.927                 12.64                 33.747                        13.10      33.934             12.62
   P2_2023_ALL20_h24_direct_openmeteo  146751        365        38.837                       0.00             -68.43        11.38             14.25             14.84               14.09               14.26       gbm_direct          33.301                 14.25                 32.422                        16.52      32.614             16.02
         P2_2024_ALL19_h1_direct_none  146669        366        15.542                        NaN            -328.46         7.11              4.01              4.63               10.69               10.94     xgb_residual          13.842                 10.94                 13.702                        11.84      13.695             11.89
    P2_2024_ALL19_h1_direct_openmeteo  146669        366        15.542                        NaN            -328.46         7.48              3.85              4.48                9.93               10.70     xgb_residual          13.880                 10.70                 13.673                        12.03      13.691             11.91
        P2_2024_ALL19_h24_direct_none  142882        366        38.166                       0.00             -76.31         9.27             12.50             12.45               11.54               11.52       xgb_direct          33.413                 12.45                 33.279                        12.80      33.278             12.81
   P2_2024_ALL19_h24_direct_openmeteo  142882        366        38.166                       0.00             -76.31         9.35             14.69             14.56               13.25               13.07       xgb_direct          32.610                 14.56                 32.551                        14.71      32.222             15.57
         P1_LOCO_Delhi_h1_direct_none  208048       2524        30.616                        NaN            -315.90         5.44            -45.13            -42.83                6.43                7.44     gbm_residual          28.648                  6.43                 27.593                         9.87      27.853              9.03
    P1_LOCO_Delhi_h1_direct_openmeteo  208048       2524        30.616                        NaN            -315.90         6.09            -40.58            -40.98                3.40                4.11     gbm_residual          29.575                  3.40                 28.161                         8.02      27.577              9.93
        P1_LOCO_Delhi_h24_direct_none  201649       2516        75.540                       0.00             -69.03         7.44              1.82              3.14                3.26                3.12       xgb_direct          73.169                  3.14                 73.736                         2.39      73.722              2.41
   P1_LOCO_Delhi_h24_direct_openmeteo  201649       2516        75.540                       0.00             -69.03         9.29              2.66              0.29                4.68                4.53       xgb_direct          75.317                  0.29                 71.595                         5.22      73.018              3.34
        P1_LOCO_Mumbai_h1_direct_none  124425       1671        14.168                        NaN            -216.11         1.83              6.29              6.64                5.02                4.63     xgb_residual          13.512                  4.63                 13.335                         5.88      13.348              5.79
   P1_LOCO_Mumbai_h1_direct_openmeteo  124425       1671        14.168                        NaN            -216.11         1.83              6.95              6.91                5.97                6.25     xgb_residual          13.282                  6.25                 13.261                         6.40      13.298              6.14
       P1_LOCO_Kolkata_h1_direct_none  155533       2030        10.982                        NaN            -382.44         9.31             10.50             10.91               11.70               11.46     xgb_residual           9.723                 11.46                  9.545                        13.08       9.535             13.17
  P1_LOCO_Kolkata_h1_direct_openmeteo  155533       2030        10.982                        NaN            -382.44         9.08              8.52              5.56                9.97                6.41     xgb_residual          10.278                  6.41                  9.754                        11.18       9.714             11.55
       P1_LOCO_Chennai_h1_direct_none  118085       2407        20.754                        NaN            -145.52         3.05              4.01              2.87                6.67                6.15     xgb_residual          19.477                  6.15                 19.280                         7.10      19.165              7.65
  P1_LOCO_Chennai_h1_direct_openmeteo  118085       2407        20.754                        NaN            -145.52         3.56              4.32              3.90                6.67                6.20     xgb_residual          19.466                  6.20                 19.718                         4.99      19.603              5.55
     P1_LOCO_Bengaluru_h1_direct_none  168738       2433        13.624                        NaN            -243.42         5.24              6.97              7.71                8.09                8.71     xgb_residual          12.437                  8.71                 12.362                         9.27      12.338              9.44
P1_LOCO_Bengaluru_h1_direct_openmeteo  168738       2433        13.624                        NaN            -243.42         2.58              7.84              6.39                9.00                8.14     xgb_residual          12.516                  8.14                 16.800                       -23.31      14.394             -5.65
```

## 1B. MAE & Skill MAE%

```text
                               run_id  persist_MAE  same_hour_yesterday_MAE  same_hour_yesterday_sMAE%  climatology_MAE  climatology_sMAE%  ridge_MAE  ridge_sMAE%  best_tree_MAE  best_tree_sMAE%  equal_neural_ens_MAE  equal_neural_ens_sMAE%  stack_MAE  stack_sMAE%
         P2_2023_ALL20_h1_direct_none        7.478                      NaN                        NaN           44.146            -490.33      7.083         5.28          6.753             9.70                 6.678                   10.71      6.645        11.14
    P2_2023_ALL20_h1_direct_openmeteo        7.478                      NaN                        NaN           44.146            -490.33      7.240         3.19          6.683            10.63                 6.595                   11.81      6.567        12.19
         P2_2023_ALL20_h6_direct_none       20.477                   18.499                       9.66           44.180            -115.76     16.166        21.05         14.745            27.99                14.699                   28.21     14.435        29.51
    P2_2023_ALL20_h6_direct_openmeteo       20.477                   18.499                       9.66           44.180            -115.76     16.611        18.88         14.360            29.87                14.104                   31.12     13.838        32.42
        P2_2023_ALL20_h12_direct_none       24.171                   18.529                      23.34           44.224             -82.96     17.760        26.52         16.691            30.95                16.483                   31.81     16.407        32.12
   P2_2023_ALL20_h12_direct_openmeteo       24.171                   18.529                      23.34           44.224             -82.96     18.110        25.08         16.028            33.69                15.648                   35.26     15.396        36.30
        P2_2023_ALL20_h24_direct_none       18.614                   18.614                       0.00           44.168            -137.28     18.289         1.75         17.769             4.54                17.396                    6.55     17.321         6.95
   P2_2023_ALL20_h24_direct_openmeteo       18.614                   18.614                       0.00           44.168            -137.28     18.814        -1.07         17.456             6.22                16.666                   10.47     16.454        11.61
         P2_2024_ALL19_h1_direct_none        6.969                      NaN                        NaN           44.364            -536.59      6.695         3.94          6.354             8.82                 6.290                    9.74      6.297         9.64
    P2_2024_ALL19_h1_direct_openmeteo        6.969                      NaN                        NaN           44.364            -536.59      6.858         1.59          6.361             8.72                 6.343                    8.98      6.356         8.80
        P2_2024_ALL19_h24_direct_none       17.272                   17.272                       0.00           44.650            -158.51     17.098         1.01         16.076             6.92                16.054                    7.05     16.031         7.19
   P2_2024_ALL19_h24_direct_openmeteo       17.272                   17.272                       0.00           44.650            -158.51     18.016        -4.31         15.879             8.06                15.932                    7.76     15.773         8.68
         P1_LOCO_Delhi_h1_direct_none       17.885                      NaN                        NaN           79.391            -343.90     17.204         3.81         16.581             7.29                15.961                   10.76     16.026        10.39
    P1_LOCO_Delhi_h1_direct_openmeteo       17.885                      NaN                        NaN           79.391            -343.90     17.030         4.78         17.155             4.08                15.987                   10.61     15.792        11.70
        P1_LOCO_Delhi_h24_direct_none       44.221                   44.221                       0.00           79.433             -79.63     41.554         6.03         42.040             4.93                42.317                    4.30     42.477         3.94
   P1_LOCO_Delhi_h24_direct_openmeteo       44.221                   44.221                       0.00           79.433             -79.63     40.777         7.79         42.834             3.14                41.047                    7.18     41.690         5.72
        P1_LOCO_Mumbai_h1_direct_none        6.599                      NaN                        NaN           37.052            -461.45      6.550         0.74          6.396             3.09                 6.383                    3.28      6.382         3.29
   P1_LOCO_Mumbai_h1_direct_openmeteo        6.599                      NaN                        NaN           37.052            -461.45      6.760        -2.44          6.413             2.83                 6.428                    2.60      6.443         2.36
       P1_LOCO_Kolkata_h1_direct_none        5.303                      NaN                        NaN           40.625            -666.11      5.082         4.16          4.974             6.21                 4.873                    8.10      4.899         7.61
  P1_LOCO_Kolkata_h1_direct_openmeteo        5.303                      NaN                        NaN           40.625            -666.11      5.271         0.60          5.352            -0.93                 5.182                    2.27      5.159         2.72
       P1_LOCO_Chennai_h1_direct_none        7.776                      NaN                        NaN           41.837            -438.04      7.728         0.61          7.385             5.02                 7.349                    5.49      7.334         5.68
  P1_LOCO_Chennai_h1_direct_openmeteo        7.776                      NaN                        NaN           41.837            -438.04      7.678         1.26          7.281             6.37                 7.395                    4.90      7.355         5.41
     P1_LOCO_Bengaluru_h1_direct_none        5.018                      NaN                        NaN           41.571            -728.40      4.989         0.58          4.659             7.16                 4.646                    7.42      4.636         7.61
P1_LOCO_Bengaluru_h1_direct_openmeteo        5.018                      NaN                        NaN           41.571            -728.40      6.030       -20.17          4.782             4.70                12.248                 -144.06      8.874       -76.83
```

## 2. Per-Family Seed Mean ± Std & Ensemble Skill

```text
                               run_id                     gru                    lstm                  bilstm                     tcn                cnn_lstm           EQ_ENS
         P2_2023_ALL20_h1_direct_none 14.720+-0.049 [+12.97%] 14.733+-0.067 [+12.89%] 14.741+-0.053 [+12.85%] 14.993+-0.057 [+11.62%] 14.604+-0.035 [+13.71%] 14.588 [+13.35%]
    P2_2023_ALL20_h1_direct_openmeteo 14.687+-0.049 [+13.54%] 14.737+-0.039 [+13.56%] 14.800+-0.106 [+13.37%] 14.988+-0.120 [+12.73%] 14.790+-0.088 [+13.52%] 14.452 [+14.16%]
         P2_2023_ALL20_h6_direct_none 29.895+-0.045 [+30.23%] 30.160+-0.060 [+29.98%] 30.020+-0.187 [+30.07%] 30.941+-0.134 [+27.79%] 29.845+-0.232 [+30.69%] 29.576 [+30.49%]
    P2_2023_ALL20_h6_direct_openmeteo 28.935+-0.149 [+33.39%] 29.522+-0.216 [+32.33%] 29.304+-0.457 [+32.79%] 29.550+-0.168 [+31.97%] 28.864+-0.132 [+33.80%] 28.165 [+33.81%]
        P2_2023_ALL20_h12_direct_none 32.950+-0.206 [+32.65%] 32.922+-0.103 [+32.89%] 32.951+-0.208 [+32.73%] 32.677+-0.255 [+33.19%] 32.813+-0.150 [+33.06%] 32.468 [+33.22%]
   P2_2023_ALL20_h12_direct_openmeteo 31.788+-0.342 [+36.09%] 31.953+-0.257 [+35.76%] 31.850+-0.312 [+35.82%] 31.570+-0.196 [+36.34%] 31.961+-0.366 [+35.59%] 30.835 [+36.58%]
        P2_2023_ALL20_h24_direct_none 33.911+-0.167 [+13.10%] 34.086+-0.101 [+12.56%] 33.980+-0.204 [+12.86%] 34.193+-0.063 [+12.43%] 34.089+-0.234 [+12.96%] 33.747 [+13.10%]
   P2_2023_ALL20_h24_direct_openmeteo 32.900+-0.110 [+16.04%] 33.347+-0.292 [+15.54%] 33.291+-0.651 [+15.89%] 33.070+-0.150 [+15.97%] 33.047+-0.152 [+16.08%] 32.422 [+16.52%]
         P2_2024_ALL19_h1_direct_none 13.832+-0.030 [+11.46%] 13.883+-0.134 [+11.40%] 13.879+-0.038 [+11.28%] 13.972+-0.084 [+10.87%] 13.886+-0.079 [+11.46%] 13.702 [+11.84%]
    P2_2024_ALL19_h1_direct_openmeteo 13.928+-0.064 [+11.49%] 13.879+-0.092 [+11.83%] 13.850+-0.059 [+12.08%]  14.307+-0.088 [+9.79%] 14.111+-0.107 [+10.82%] 13.673 [+12.03%]
        P2_2024_ALL19_h24_direct_none 33.426+-0.164 [+12.71%] 33.364+-0.075 [+12.92%] 33.583+-0.055 [+12.32%] 33.648+-0.079 [+12.22%] 33.529+-0.183 [+12.61%] 33.279 [+12.80%]
   P2_2024_ALL19_h24_direct_openmeteo 33.133+-0.283 [+13.96%] 33.132+-0.454 [+14.52%] 33.051+-0.312 [+14.68%] 33.326+-0.451 [+13.74%] 33.297+-0.208 [+13.76%] 32.551 [+14.71%]
         P1_LOCO_Delhi_h1_direct_none 27.917+-0.361 [+10.07%]  28.902+-0.513 [+7.25%]  29.990+-1.267 [+5.83%]  28.458+-0.268 [+7.93%]  27.967+-0.130 [+9.84%]  27.593 [+9.87%]
    P1_LOCO_Delhi_h1_direct_openmeteo  28.543+-0.430 [+7.99%]  29.646+-0.588 [+4.18%]  30.297+-0.921 [+2.46%]  28.297+-0.583 [+9.20%]  28.244+-0.486 [+9.06%]  28.161 [+8.02%]
        P1_LOCO_Delhi_h24_direct_none  70.998+-1.001 [+6.24%]  79.692+-3.025 [-4.58%]  79.506+-4.148 [-4.19%]  72.037+-1.081 [+4.88%]  74.808+-1.715 [+2.32%]  73.736 [+2.39%]
   P1_LOCO_Delhi_h24_direct_openmeteo  72.198+-1.994 [+4.83%]  74.389+-4.849 [+3.10%]  72.604+-1.378 [+4.47%]  72.754+-3.910 [+4.91%]  71.815+-1.411 [+5.79%]  71.595 [+5.22%]
        P1_LOCO_Mumbai_h1_direct_none  13.321+-0.011 [+6.29%]  13.457+-0.035 [+5.27%]  13.431+-0.162 [+5.57%]  13.665+-0.049 [+4.12%]  13.422+-0.109 [+5.83%]  13.335 [+5.88%]
   P1_LOCO_Mumbai_h1_direct_openmeteo  13.411+-0.084 [+6.19%]  13.456+-0.085 [+5.83%]  13.450+-0.096 [+6.11%]  13.802+-0.081 [+4.29%]  13.525+-0.192 [+6.12%]  13.261 [+6.40%]
       P1_LOCO_Kolkata_h1_direct_none  9.701+-0.037 [+12.28%]  9.708+-0.094 [+12.38%]  9.688+-0.024 [+12.40%]  9.824+-0.089 [+11.68%]  9.675+-0.044 [+12.94%]  9.545 [+13.08%]
  P1_LOCO_Kolkata_h1_direct_openmeteo  10.140+-0.154 [+9.72%] 10.137+-0.141 [+10.27%]  10.151+-0.148 [+9.74%]  10.459+-0.335 [+9.10%]  10.273+-0.235 [+9.84%]  9.754 [+11.18%]
       P1_LOCO_Chennai_h1_direct_none  19.267+-0.055 [+7.49%]  19.460+-0.093 [+6.53%]  19.403+-0.063 [+6.77%]  19.819+-0.315 [+5.58%]  19.393+-0.136 [+6.97%]  19.280 [+7.10%]
  P1_LOCO_Chennai_h1_direct_openmeteo  20.175+-0.338 [+3.41%]  20.101+-0.283 [+3.68%]  19.649+-0.101 [+5.69%]  20.087+-0.139 [+4.04%]  19.725+-0.126 [+5.65%]  19.718 [+4.99%]
     P1_LOCO_Bengaluru_h1_direct_none  12.433+-0.035 [+9.18%]  12.437+-0.015 [+8.95%]  12.446+-0.049 [+9.03%]  12.688+-0.060 [+7.43%]  12.481+-0.054 [+9.19%]  12.362 [+9.27%]
P1_LOCO_Bengaluru_h1_direct_openmeteo 18.096+-3.230 [-22.98%] 22.871+-7.071 [-53.42%] 20.963+-4.317 [-45.48%]  16.299+-1.981 [+1.83%] 17.954+-4.042 [-22.94%] 16.800 [-23.31%]
```

## 3A. DATE-Block 95% CIs (Within-Run)

```text
                               run_id                A                   B  n_blocks  dRMSE_mean  dRMSE_lo  dRMSE_hi  dMAE_mean  dMAE_lo  dMAE_hi
         P2_2023_ALL20_h1_direct_none equal_neural_ens         persistence       365      -2.248    -2.530    -1.962     -0.801   -0.886   -0.717
         P2_2023_ALL20_h1_direct_none equal_neural_ens           best_tree       365      -0.344    -0.592    -0.115     -0.075   -0.100   -0.051
         P2_2023_ALL20_h1_direct_none            stack    equal_neural_ens       365      -0.027    -0.087     0.034     -0.032   -0.044   -0.022
    P2_2023_ALL20_h1_direct_openmeteo equal_neural_ens         persistence       365      -2.391    -2.710    -2.037     -0.884   -0.978   -0.793
    P2_2023_ALL20_h1_direct_openmeteo equal_neural_ens           best_tree       365      -0.303    -0.502    -0.128     -0.089   -0.129   -0.054
    P2_2023_ALL20_h1_direct_openmeteo            stack    equal_neural_ens       365       0.064    -0.014     0.151     -0.028   -0.040   -0.015
         P2_2023_ALL20_h6_direct_none equal_neural_ens         persistence       365     -12.992   -14.473   -11.505     -5.778   -6.342   -5.249
         P2_2023_ALL20_h6_direct_none equal_neural_ens           best_tree       365      -0.122    -0.359     0.115     -0.047   -0.130    0.034
         P2_2023_ALL20_h6_direct_none equal_neural_ens same_hour_yesterday       365      -8.926   -10.486    -7.428     -3.803   -4.283   -3.350
         P2_2023_ALL20_h6_direct_none            stack    equal_neural_ens       365      -0.206    -0.306    -0.098     -0.264   -0.306   -0.224
    P2_2023_ALL20_h6_direct_openmeteo equal_neural_ens         persistence       365     -14.399   -16.010   -12.709     -6.372   -6.991   -5.776
    P2_2023_ALL20_h6_direct_openmeteo equal_neural_ens           best_tree       365      -0.695    -1.111    -0.325     -0.260   -0.398   -0.131
    P2_2023_ALL20_h6_direct_openmeteo equal_neural_ens same_hour_yesterday       365     -10.334   -11.977    -8.753     -4.397   -4.943   -3.890
    P2_2023_ALL20_h6_direct_openmeteo            stack    equal_neural_ens       365      -0.210    -0.338    -0.092     -0.264   -0.311   -0.219
        P2_2023_ALL20_h12_direct_none equal_neural_ens         persistence       365     -16.142   -18.192   -14.048     -7.671   -8.450   -6.928
        P2_2023_ALL20_h12_direct_none equal_neural_ens           best_tree       365      -0.067    -0.306     0.176     -0.208   -0.294   -0.121
        P2_2023_ALL20_h12_direct_none equal_neural_ens same_hour_yesterday       365      -6.072    -7.431    -4.775     -2.049   -2.400   -1.697
        P2_2023_ALL20_h12_direct_none            stack    equal_neural_ens       365      -0.127    -0.260    -0.004     -0.075   -0.118   -0.030
   P2_2023_ALL20_h12_direct_openmeteo equal_neural_ens         persistence       365     -17.784   -20.055   -15.533     -8.505   -9.375   -7.679
   P2_2023_ALL20_h12_direct_openmeteo equal_neural_ens           best_tree       365      -0.719    -1.278    -0.202     -0.381   -0.534   -0.220
   P2_2023_ALL20_h12_direct_openmeteo equal_neural_ens same_hour_yesterday       365      -7.714    -9.247    -6.206     -2.883   -3.316   -2.436
   P2_2023_ALL20_h12_direct_openmeteo            stack    equal_neural_ens       365      -0.191    -0.364    -0.031     -0.251   -0.305   -0.199
        P2_2023_ALL20_h24_direct_none equal_neural_ens         persistence       365      -5.092    -6.465    -3.795     -1.217   -1.511   -0.915
        P2_2023_ALL20_h24_direct_none equal_neural_ens           best_tree       365      -0.179    -0.324    -0.045     -0.374   -0.457   -0.286
        P2_2023_ALL20_h24_direct_none equal_neural_ens same_hour_yesterday       365      -5.092    -6.465    -3.795     -1.217   -1.511   -0.915
        P2_2023_ALL20_h24_direct_none            stack    equal_neural_ens       365       0.187    -0.010     0.368     -0.074   -0.150   -0.005
   P2_2023_ALL20_h24_direct_openmeteo equal_neural_ens         persistence       365      -6.421    -7.876    -4.945     -1.947   -2.357   -1.553
   P2_2023_ALL20_h24_direct_openmeteo equal_neural_ens           best_tree       365      -0.886    -1.380    -0.405     -0.795   -0.970   -0.612
   P2_2023_ALL20_h24_direct_openmeteo equal_neural_ens same_hour_yesterday       365      -6.421    -7.876    -4.945     -1.947   -2.357   -1.553
   P2_2023_ALL20_h24_direct_openmeteo            stack    equal_neural_ens       365       0.191    -0.058     0.430     -0.211   -0.300   -0.125
         P1_LOCO_Delhi_h1_direct_none equal_neural_ens         persistence      2524      -3.034    -3.396    -2.629     -1.928   -2.061   -1.791
         P1_LOCO_Delhi_h1_direct_none equal_neural_ens           best_tree      2524      -1.062    -1.350    -0.724     -0.624   -0.711   -0.527
         P1_LOCO_Delhi_h1_direct_none            stack    equal_neural_ens      2524       0.256     0.143     0.387      0.065    0.041    0.091
    P1_LOCO_Delhi_h1_direct_openmeteo equal_neural_ens         persistence      2524      -2.468    -2.975    -1.910     -1.902   -2.061   -1.743
    P1_LOCO_Delhi_h1_direct_openmeteo equal_neural_ens           best_tree      2524      -1.427    -1.879    -0.909     -1.173   -1.311   -1.024
    P1_LOCO_Delhi_h1_direct_openmeteo            stack    equal_neural_ens      2524      -0.581    -0.712    -0.467     -0.194   -0.239   -0.151
        P1_LOCO_Delhi_h24_direct_none equal_neural_ens         persistence      2516      -1.781    -4.465     0.669     -1.893   -2.826   -0.954
        P1_LOCO_Delhi_h24_direct_none equal_neural_ens           best_tree      2516       0.562     0.143     0.961      0.273    0.056    0.489
        P1_LOCO_Delhi_h24_direct_none equal_neural_ens same_hour_yesterday      2516      -1.781    -4.465     0.669     -1.893   -2.826   -0.954
        P1_LOCO_Delhi_h24_direct_none            stack    equal_neural_ens      2516      -0.006    -0.298     0.282      0.162    0.030    0.294
   P1_LOCO_Delhi_h24_direct_openmeteo equal_neural_ens         persistence      2516      -3.920    -6.489    -1.564     -3.164   -4.077   -2.260
   P1_LOCO_Delhi_h24_direct_openmeteo equal_neural_ens           best_tree      2516      -3.720    -4.327    -3.153     -1.788   -2.101   -1.482
   P1_LOCO_Delhi_h24_direct_openmeteo equal_neural_ens same_hour_yesterday      2516      -3.920    -6.489    -1.564     -3.164   -4.077   -2.260
   P1_LOCO_Delhi_h24_direct_openmeteo            stack    equal_neural_ens      2516       1.423     1.209     1.630      0.642    0.518    0.763
         P2_2024_ALL19_h1_direct_none equal_neural_ens         persistence       366      -1.833    -2.159    -1.523     -0.677   -0.750   -0.605
         P2_2024_ALL19_h1_direct_none equal_neural_ens           best_tree       366      -0.139    -0.253    -0.044     -0.064   -0.089   -0.042
         P2_2024_ALL19_h1_direct_none            stack    equal_neural_ens       366      -0.007    -0.018     0.004      0.007    0.003    0.010
    P2_2024_ALL19_h1_direct_openmeteo equal_neural_ens         persistence       366      -1.863    -2.169    -1.562     -0.624   -0.702   -0.549
    P2_2024_ALL19_h1_direct_openmeteo equal_neural_ens           best_tree       366      -0.204    -0.344    -0.070     -0.018   -0.055    0.016
    P2_2024_ALL19_h1_direct_openmeteo            stack    equal_neural_ens       366       0.019    -0.000     0.038      0.013    0.006    0.020
        P2_2024_ALL19_h24_direct_none equal_neural_ens         persistence       366      -4.914    -7.233    -2.952     -1.215   -1.584   -0.862
        P2_2024_ALL19_h24_direct_none equal_neural_ens           best_tree       366      -0.131    -0.410     0.148     -0.022   -0.111    0.076
        P2_2024_ALL19_h24_direct_none equal_neural_ens same_hour_yesterday       366      -4.914    -7.233    -2.952     -1.215   -1.584   -0.862
        P2_2024_ALL19_h24_direct_none            stack    equal_neural_ens       366      -0.004    -0.072     0.064     -0.023   -0.047   -0.002
   P2_2024_ALL19_h24_direct_openmeteo equal_neural_ens         persistence       366      -5.646    -7.828    -3.473     -1.342   -1.785   -0.890
   P2_2024_ALL19_h24_direct_openmeteo equal_neural_ens           best_tree       366      -0.062    -0.671     0.569      0.052   -0.136    0.245
   P2_2024_ALL19_h24_direct_openmeteo equal_neural_ens same_hour_yesterday       366      -5.646    -7.828    -3.473     -1.342   -1.785   -0.890
   P2_2024_ALL19_h24_direct_openmeteo            stack    equal_neural_ens       366      -0.328    -0.542    -0.126     -0.159   -0.222   -0.102
        P1_LOCO_Mumbai_h1_direct_none equal_neural_ens         persistence      1671      -0.824    -1.110    -0.573     -0.215   -0.248   -0.180
        P1_LOCO_Mumbai_h1_direct_none equal_neural_ens           best_tree      1671      -0.175    -0.495     0.074     -0.013   -0.034    0.008
        P1_LOCO_Mumbai_h1_direct_none            stack    equal_neural_ens      1671       0.012    -0.028     0.058     -0.001   -0.007    0.006
   P1_LOCO_Mumbai_h1_direct_openmeteo equal_neural_ens         persistence      1671      -0.899    -1.170    -0.650     -0.171   -0.206   -0.135
   P1_LOCO_Mumbai_h1_direct_openmeteo equal_neural_ens           best_tree      1671      -0.021    -0.160     0.115      0.015   -0.004    0.032
   P1_LOCO_Mumbai_h1_direct_openmeteo            stack    equal_neural_ens      1671       0.036    -0.019     0.099      0.016    0.008    0.025
       P1_LOCO_Kolkata_h1_direct_none equal_neural_ens         persistence      2030      -1.434    -1.635    -1.248     -0.429   -0.490   -0.364
       P1_LOCO_Kolkata_h1_direct_none equal_neural_ens           best_tree      2030      -0.177    -0.277    -0.091     -0.100   -0.119   -0.083
       P1_LOCO_Kolkata_h1_direct_none            stack    equal_neural_ens      2030      -0.010    -0.043     0.020      0.026    0.020    0.033
  P1_LOCO_Kolkata_h1_direct_openmeteo equal_neural_ens         persistence      2030      -1.226    -1.467    -1.021     -0.121   -0.184   -0.063
  P1_LOCO_Kolkata_h1_direct_openmeteo equal_neural_ens           best_tree      2030      -0.524    -0.671    -0.396     -0.170   -0.212   -0.132
  P1_LOCO_Kolkata_h1_direct_openmeteo            stack    equal_neural_ens      2030      -0.041    -0.070    -0.000     -0.023   -0.034   -0.014
       P1_LOCO_Chennai_h1_direct_none equal_neural_ens         persistence      2407      -1.474    -1.927    -1.001     -0.427   -0.486   -0.366
       P1_LOCO_Chennai_h1_direct_none equal_neural_ens           best_tree      2407      -0.203    -0.461     0.120     -0.037   -0.071   -0.006
       P1_LOCO_Chennai_h1_direct_none            stack    equal_neural_ens      2407      -0.114    -0.209    -0.037     -0.015   -0.024   -0.007
  P1_LOCO_Chennai_h1_direct_openmeteo equal_neural_ens         persistence      2407      -1.034    -1.595    -0.403     -0.381   -0.444   -0.316
  P1_LOCO_Chennai_h1_direct_openmeteo equal_neural_ens           best_tree      2407       0.250    -0.088     0.658      0.114    0.074    0.154
  P1_LOCO_Chennai_h1_direct_openmeteo            stack    equal_neural_ens      2407      -0.116    -0.166    -0.069     -0.040   -0.048   -0.032
     P1_LOCO_Bengaluru_h1_direct_none equal_neural_ens         persistence      2433      -1.260    -1.458    -1.082     -0.373   -0.404   -0.344
     P1_LOCO_Bengaluru_h1_direct_none equal_neural_ens           best_tree      2433      -0.074    -0.279     0.099     -0.013   -0.028   -0.000
     P1_LOCO_Bengaluru_h1_direct_none            stack    equal_neural_ens      2433      -0.024    -0.071     0.018     -0.010   -0.014   -0.006
P1_LOCO_Bengaluru_h1_direct_openmeteo equal_neural_ens         persistence      2433       3.186     2.796     3.588      7.228    7.105    7.346
P1_LOCO_Bengaluru_h1_direct_openmeteo equal_neural_ens           best_tree      2433       4.289     3.971     4.630      7.464    7.351    7.576
P1_LOCO_Bengaluru_h1_direct_openmeteo            stack    equal_neural_ens      2433      -2.408    -2.519    -2.282     -3.373   -3.430   -3.316
```

## 3B. DATE-Block 95% CIs (Open-Meteo vs None)

```text
protocol   heldout  horizon  N_test  n_blocks  dRMSE_mean  dRMSE_lo  dRMSE_hi  dMAE_mean  dMAE_lo  dMAE_hi
 P1_LOCO Bengaluru        1  168738      2433       4.446     4.143     4.768      7.600    7.488    7.713
 P1_LOCO   Chennai        1  118085      2407       0.440     0.127     0.809      0.046    0.015    0.078
 P1_LOCO     Delhi        1  208048      2524       0.566     0.331     0.816      0.027   -0.038    0.091
 P1_LOCO     Delhi       24  201649      2516      -2.139    -2.717    -1.562     -1.271   -1.582   -0.969
 P1_LOCO   Kolkata        1  155533      2030       0.208     0.133     0.282      0.308    0.273    0.342
 P1_LOCO    Mumbai        1  124425      1671      -0.075    -0.128    -0.017      0.044    0.028    0.061
 P2_2023     ALL20        1  151279       365      -0.142    -0.308     0.115     -0.083   -0.107   -0.059
 P2_2023     ALL20        6  149831       365      -1.407    -1.865    -0.962     -0.594   -0.746   -0.445
 P2_2023     ALL20       12  148395       365      -1.642    -2.272    -0.945     -0.834   -1.067   -0.621
 P2_2023     ALL20       24  146751       365      -1.329    -2.017    -0.624     -0.730   -0.972   -0.506
 P2_2024     ALL19        1  146669       366      -0.030    -0.124     0.071      0.053    0.023    0.082
 P2_2024     ALL19       24  142882       366      -0.732    -1.588     0.123     -0.127   -0.394    0.134
```

## 4. Regime Tables (All 24 Runs)

### `P2_2023_ALL20_h1_direct_none`
```text
                  regime            model      N   RMSE    MAE  skill_RMSE_pct  skill_MAE_pct
               y_t >= 60        best_tree  41978  26.14  13.70           11.99          13.40
               y_t >= 60      climatology  41978 106.36  65.54         -258.08        -314.20
               y_t >= 60 equal_neural_ens  41978  25.41  13.41           14.44          15.27
               y_t >= 60      persistence  41978  29.70  15.82            0.00           0.00
               y_t >= 60            ridge  41978  27.22  14.22            8.37          10.12
               y_t >= 60            stack  41978  25.39  13.38           14.50          15.46
              y_t >= 120        best_tree  14427  39.56  22.83           12.37          14.67
              y_t >= 120      climatology  14427 176.41 146.11         -290.78        -446.09
              y_t >= 120 equal_neural_ens  14427  38.31  22.23           15.14          16.92
              y_t >= 120      persistence  14427  45.14  26.76            0.00           0.00
              y_t >= 120            ridge  14427  41.27  23.89            8.59          10.73
              y_t >= 120            stack  14427  38.31  22.24           15.15          16.88
              y_t >= 250        best_tree   3797  60.55  36.57           12.33          13.57
              y_t >= 250      climatology   3797 287.00 268.73         -315.57        -535.09
              y_t >= 250 equal_neural_ens   3797  57.83  35.09           16.26          17.08
              y_t >= 250      persistence   3797  69.06  42.31            0.00           0.00
              y_t >= 250            ridge   3797  63.39  38.10            8.20           9.95
              y_t >= 250            stack   3797  57.76  35.13           16.36          16.99
onset (y_t - y_t-3 > 30)        best_tree   9114  43.29  24.62           11.28          13.58
onset (y_t - y_t-3 > 30)      climatology   9114 165.15 116.08         -238.47        -307.42
onset (y_t - y_t-3 > 30) equal_neural_ens   9114  41.81  23.95           14.31          15.96
onset (y_t - y_t-3 > 30)      persistence   9114  48.79  28.49            0.00           0.00
onset (y_t - y_t-3 > 30)            ridge   9114  45.40  25.73            6.95           9.70
onset (y_t - y_t-3 > 30)            stack   9114  41.70  23.88           14.54          16.18
        winter (Nov-Feb)        best_tree  51782  20.72   9.53           12.34          11.77
        winter (Nov-Feb)      climatology  51782  93.30  57.04         -294.75        -428.38
        winter (Nov-Feb) equal_neural_ens  51782  20.15   9.33           14.73          13.61
        winter (Nov-Feb)      persistence  51782  23.63  10.80            0.00           0.00
        winter (Nov-Feb)            ridge  51782  21.87  10.06            7.46           6.80
        winter (Nov-Feb)            stack  51782  20.12   9.29           14.87          13.92
  other months (Mar-Oct)        best_tree  99497  10.76   5.31            9.17           7.69
  other months (Mar-Oct)      climatology  99497  44.48  37.43         -275.60        -550.83
  other months (Mar-Oct) equal_neural_ens  99497  10.59   5.30           10.56           7.87
  other months (Mar-Oct)      persistence  99497  11.84   5.75            0.00           0.00
  other months (Mar-Oct)            ridge  99497  11.01   5.53            7.01           3.80
  other months (Mar-Oct)            stack  99497  10.57   5.27           10.75           8.42
                     all        best_tree 151279  14.93   6.75           11.30           9.70
                     all      climatology 151279  65.43  44.15         -288.62        -490.33
                     all equal_neural_ens 151279  14.59   6.68           13.35          10.71
                     all      persistence 151279  16.84   7.48            0.00           0.00
                     all            ridge 151279  15.60   7.08            7.31           5.28
                     all            stack 151279  14.56   6.65           13.51          11.14
```

### `P2_2023_ALL20_h1_direct_openmeteo`
```text
                  regime            model      N   RMSE    MAE  skill_RMSE_pct  skill_MAE_pct
               y_t >= 60        best_tree  41978  25.80  13.49           13.13          14.73
               y_t >= 60      climatology  41978 106.36  65.54         -258.08        -314.20
               y_t >= 60 equal_neural_ens  41978  25.19  13.21           15.20          16.54
               y_t >= 60      persistence  41978  29.70  15.82            0.00           0.00
               y_t >= 60            ridge  41978  26.95  14.11            9.26          10.80
               y_t >= 60            stack  41978  25.37  13.20           14.59          16.58
              y_t >= 120        best_tree  14427  39.12  22.52           13.34          15.85
              y_t >= 120      climatology  14427 176.41 146.11         -290.78        -446.09
              y_t >= 120 equal_neural_ens  14427  38.04  21.86           15.73          18.31
              y_t >= 120      persistence  14427  45.14  26.76            0.00           0.00
              y_t >= 120            ridge  14427  40.89  23.57            9.42          11.93
              y_t >= 120            stack  14427  38.42  21.94           14.89          18.00
              y_t >= 250        best_tree   3797  59.92  36.24           13.23          14.35
              y_t >= 250      climatology   3797 287.00 268.73         -315.57        -535.09
              y_t >= 250 equal_neural_ens   3797  57.62  34.55           16.56          18.35
              y_t >= 250      persistence   3797  69.06  42.31            0.00           0.00
              y_t >= 250            ridge   3797  62.97  37.88            8.82          10.48
              y_t >= 250            stack   3797  58.47  34.97           15.34          17.35
onset (y_t - y_t-3 > 30)        best_tree   9114  42.86  24.12           12.15          15.35
onset (y_t - y_t-3 > 30)      climatology   9114 165.15 116.08         -238.47        -307.42
onset (y_t - y_t-3 > 30) equal_neural_ens   9114  41.47  23.41           15.00          17.85
onset (y_t - y_t-3 > 30)      persistence   9114  48.79  28.49            0.00           0.00
onset (y_t - y_t-3 > 30)            ridge   9114  44.99  25.42            7.79          10.80
onset (y_t - y_t-3 > 30)            stack   9114  41.92  23.50           14.09          17.53
        winter (Nov-Feb)        best_tree  51782  20.53   9.44           13.16          12.59
        winter (Nov-Feb)      climatology  51782  93.30  57.04         -294.75        -428.38
        winter (Nov-Feb) equal_neural_ens  51782  19.97   9.18           15.49          14.95
        winter (Nov-Feb)      persistence  51782  23.63  10.80            0.00           0.00
        winter (Nov-Feb)            ridge  51782  21.70  10.10            8.18           6.48
        winter (Nov-Feb)            stack  51782  20.11   9.17           14.90          15.06
  other months (Mar-Oct)        best_tree  99497  10.57   5.25           10.78           8.72
  other months (Mar-Oct)      climatology  99497  44.48  37.43         -275.60        -550.83
  other months (Mar-Oct) equal_neural_ens  99497  10.49   5.25           11.46           8.74
  other months (Mar-Oct)      persistence  99497  11.84   5.75            0.00           0.00
  other months (Mar-Oct)            ridge  99497  11.03   5.75            6.88          -0.03
  other months (Mar-Oct)            stack  99497  10.48   5.21           11.49           9.39
                     all        best_tree 151279  14.75   6.68           12.37          10.63
                     all      climatology 151279  65.43  44.15         -288.62        -490.33
                     all equal_neural_ens 151279  14.45   6.60           14.16          11.81
                     all      persistence 151279  16.84   7.48            0.00           0.00
                     all            ridge 151279  15.53   7.24            7.76           3.19
                     all            stack 151279  14.52   6.57           13.77          12.19
```

### `P2_2023_ALL20_h6_direct_none`
```text
                  regime               model      N   RMSE    MAE  skill_RMSE_pct  skill_MAE_pct
               y_t >= 60           best_tree  41695  48.89  29.25           32.70          34.43
               y_t >= 60         climatology  41695 104.00  63.34          -43.19         -41.99
               y_t >= 60    equal_neural_ens  41695  48.57  29.04           33.13          34.89
               y_t >= 60         persistence  41695  72.63  44.61            0.00           0.00
               y_t >= 60               ridge  41695  51.42  30.89           29.21          30.76
               y_t >= 60 same_hour_yesterday  41695  62.99  36.94           13.27          17.19
               y_t >= 60               stack  41695  48.26  28.63           33.56          35.82
              y_t >= 120           best_tree  14336  69.83  46.65           34.87          38.90
              y_t >= 120         climatology  14336 163.86 125.80          -52.83         -64.77
              y_t >= 120    equal_neural_ens  14336  69.24  45.96           35.42          39.80
              y_t >= 120         persistence  14336 107.21  76.35            0.00           0.00
              y_t >= 120               ridge  14336  73.44  49.96           31.50          34.56
              y_t >= 120 same_hour_yesterday  14336  89.49  59.39           16.53          22.22
              y_t >= 120               stack  14336  68.82  45.53           35.81          40.36
              y_t >= 250           best_tree   3766  95.37  68.70           38.80          42.19
              y_t >= 250         climatology   3766 237.61 207.02          -52.47         -74.22
              y_t >= 250    equal_neural_ens   3766  93.88  66.81           39.76          43.77
              y_t >= 250         persistence   3766 155.84 118.82            0.00           0.00
              y_t >= 250               ridge   3766 100.29  74.05           35.64          37.68
              y_t >= 250 same_hour_yesterday   3766 122.21  87.51           21.58          26.35
              y_t >= 250               stack   3766  93.31  66.49           40.12          44.04
onset (y_t - y_t-3 > 30)           best_tree   9068  64.87  41.13           36.31          39.44
onset (y_t - y_t-3 > 30)         climatology   9068 141.36  95.37          -38.78         -40.41
onset (y_t - y_t-3 > 30)    equal_neural_ens   9068  64.70  40.61           36.48          40.21
onset (y_t - y_t-3 > 30)         persistence   9068 101.86  67.92            0.00           0.00
onset (y_t - y_t-3 > 30)               ridge   9068  67.83  43.81           33.40          35.50
onset (y_t - y_t-3 > 30) same_hour_yesterday   9068  82.11  51.34           19.39          24.41
onset (y_t - y_t-3 > 30)               stack   9068  64.23  40.26           36.95          40.73
        winter (Nov-Feb)           best_tree  51365  42.58  22.68           29.97          29.72
        winter (Nov-Feb)         climatology  51365  93.18  57.04          -53.25         -76.76
        winter (Nov-Feb)    equal_neural_ens  51365  42.32  22.54           30.40          30.16
        winter (Nov-Feb)         persistence  51365  60.80  32.27            0.00           0.00
        winter (Nov-Feb)               ridge  51365  44.70  24.54           26.48          23.96
        winter (Nov-Feb) same_hour_yesterday  51365  55.40  29.26            8.89           9.35
        winter (Nov-Feb)               stack  51365  42.03  22.20           30.88          31.21
  other months (Mar-Oct)           best_tree  98466  19.89  10.61           30.81          25.96
  other months (Mar-Oct)         climatology  98466  44.54  37.47          -54.89        -161.59
  other months (Mar-Oct)    equal_neural_ens  98466  19.92  10.61           30.71          25.93
  other months (Mar-Oct)         persistence  98466  28.75  14.32            0.00           0.00
  other months (Mar-Oct)               ridge  98466  21.42  11.80           25.49          17.63
  other months (Mar-Oct) same_hour_yesterday  98466  25.53  12.89           11.19          10.03
  other months (Mar-Oct)               stack  98466  19.78  10.38           31.22          27.51
                     all           best_tree 149831  29.69  14.75           30.22          27.99
                     all         climatology 149831  65.42  44.18          -53.74        -115.76
                     all    equal_neural_ens 149831  29.58  14.70           30.49          28.21
                     all         persistence 149831  42.55  20.48            0.00           0.00
                     all               ridge 149831  31.41  16.17           26.18          21.05
                     all same_hour_yesterday 149831  38.48  18.50            9.57           9.66
                     all               stack 149831  29.37  14.43           30.98          29.51
```

### `P2_2023_ALL20_h6_direct_openmeteo`
```text
                  regime               model      N   RMSE    MAE  skill_RMSE_pct  skill_MAE_pct
               y_t >= 60           best_tree  41695  47.73  28.55           34.29          36.00
               y_t >= 60         climatology  41695 104.00  63.34          -43.19         -41.99
               y_t >= 60    equal_neural_ens  41695  46.30  27.76           36.26          37.78
               y_t >= 60         persistence  41695  72.63  44.61            0.00           0.00
               y_t >= 60               ridge  41695  50.15  30.51           30.96          31.62
               y_t >= 60 same_hour_yesterday  41695  62.99  36.94           13.27          17.19
               y_t >= 60               stack  41695  46.02  27.36           36.64          38.66
              y_t >= 120           best_tree  14336  68.79  45.78           35.84          40.04
              y_t >= 120         climatology  14336 163.86 125.80          -52.83         -64.77
              y_t >= 120    equal_neural_ens  14336  66.13  44.11           38.32          42.23
              y_t >= 120         persistence  14336 107.21  76.35            0.00           0.00
              y_t >= 120               ridge  14336  71.84  48.76           32.99          36.14
              y_t >= 120 same_hour_yesterday  14336  89.49  59.39           16.53          22.22
              y_t >= 120               stack  14336  65.95  43.66           38.49          42.82
              y_t >= 250           best_tree   3766  94.68  68.02           39.24          42.76
              y_t >= 250         climatology   3766 237.61 207.02          -52.47         -74.22
              y_t >= 250    equal_neural_ens   3766  89.83  64.41           42.36          45.80
              y_t >= 250         persistence   3766 155.84 118.82            0.00           0.00
              y_t >= 250               ridge   3766  98.78  72.94           36.61          38.61
              y_t >= 250 same_hour_yesterday   3766 122.21  87.51           21.58          26.35
              y_t >= 250               stack   3766  90.12  64.44           42.17          45.77
onset (y_t - y_t-3 > 30)           best_tree   9068  63.44  39.93           37.72          41.22
onset (y_t - y_t-3 > 30)         climatology   9068 141.36  95.37          -38.78         -40.41
onset (y_t - y_t-3 > 30)    equal_neural_ens   9068  61.06  38.26           40.05          43.68
onset (y_t - y_t-3 > 30)         persistence   9068 101.86  67.92            0.00           0.00
onset (y_t - y_t-3 > 30)               ridge   9068  66.22  43.03           34.98          36.65
onset (y_t - y_t-3 > 30) same_hour_yesterday   9068  82.11  51.34           19.39          24.41
onset (y_t - y_t-3 > 30)               stack   9068  60.86  37.95           40.25          44.13
        winter (Nov-Feb)           best_tree  51365  41.36  22.03           31.98          31.75
        winter (Nov-Feb)         climatology  51365  93.18  57.04          -53.25         -76.76
        winter (Nov-Feb)    equal_neural_ens  51365  40.16  21.46           33.95          33.50
        winter (Nov-Feb)         persistence  51365  60.80  32.27            0.00           0.00
        winter (Nov-Feb)               ridge  51365  43.67  24.25           28.18          24.86
        winter (Nov-Feb) same_hour_yesterday  51365  55.40  29.26            8.89           9.35
        winter (Nov-Feb)               stack  51365  39.94  21.13           34.31          34.51
  other months (Mar-Oct)           best_tree  98466  19.34  10.36           32.74          27.66
  other months (Mar-Oct)         climatology  98466  44.54  37.47          -54.89        -161.59
  other months (Mar-Oct)    equal_neural_ens  98466  19.13  10.27           33.48          28.33
  other months (Mar-Oct)         persistence  98466  28.75  14.32            0.00           0.00
  other months (Mar-Oct)               ridge  98466  21.37  12.63           25.67          11.85
  other months (Mar-Oct) same_hour_yesterday  98466  25.53  12.89           11.19          10.03
  other months (Mar-Oct)               stack  98466  18.89  10.03           34.31          29.96
                     all           best_tree 149831  28.85  14.36           32.20          29.87
                     all         climatology 149831  65.42  44.18          -53.74        -115.76
                     all    equal_neural_ens 149831  28.16  14.10           33.81          31.12
                     all         persistence 149831  42.55  20.48            0.00           0.00
                     all               ridge 149831  30.89  16.61           27.42          18.88
                     all same_hour_yesterday 149831  38.48  18.50            9.57           9.66
                     all               stack 149831  27.95  13.84           34.31          32.42
```

### `P2_2023_ALL20_h12_direct_none`
```text
                  regime               model      N   RMSE    MAE  skill_RMSE_pct  skill_MAE_pct
               y_t >= 60           best_tree  41397  52.16  31.53           36.36          39.91
               y_t >= 60         climatology  41397 102.51  62.06          -25.06         -18.26
               y_t >= 60    equal_neural_ens  41397  52.00  31.56           36.56          39.86
               y_t >= 60         persistence  41397  81.96  52.47            0.00           0.00
               y_t >= 60               ridge  41397  54.01  32.80           34.11          37.49
               y_t >= 60 same_hour_yesterday  41397  61.80  36.13           24.60          31.15
               y_t >= 60               stack  41397  51.80  31.19           36.80          40.55
              y_t >= 120           best_tree  14235  72.18  48.00           40.43          46.83
              y_t >= 120         climatology  14235 157.98 116.13          -30.37         -28.64
              y_t >= 120    equal_neural_ens  14235  71.92  48.03           40.65          46.79
              y_t >= 120         persistence  14235 121.18  90.28            0.00           0.00
              y_t >= 120               ridge  14235  75.14  51.00           37.99          43.51
              y_t >= 120 same_hour_yesterday  14235  85.61  55.32           29.35          38.72
              y_t >= 120               stack  14235  71.61  47.59           40.91          47.28
              y_t >= 250           best_tree   3698  96.35  69.91           45.05          49.08
              y_t >= 250         climatology   3698 222.90 189.59          -27.13         -38.10
              y_t >= 250    equal_neural_ens   3698  95.03  68.93           45.80          49.79
              y_t >= 250         persistence   3698 175.34 137.29            0.00           0.00
              y_t >= 250               ridge   3698  99.55  74.52           43.22          45.72
              y_t >= 250 same_hour_yesterday   3698 113.09  81.79           35.50          40.42
              y_t >= 250               stack   3698  94.88  68.80           45.89          49.89
onset (y_t - y_t-3 > 30)           best_tree   8997  63.42  39.26           43.95          49.40
onset (y_t - y_t-3 > 30)         climatology   8997 128.41  82.31          -13.49          -6.08
onset (y_t - y_t-3 > 30)    equal_neural_ens   8997  62.80  39.06           44.49          49.67
onset (y_t - y_t-3 > 30)         persistence   8997 113.15  77.60            0.00           0.00
onset (y_t - y_t-3 > 30)               ridge   8997  66.79  42.10           40.97          45.74
onset (y_t - y_t-3 > 30) same_hour_yesterday   8997  74.97  45.32           33.74          41.59
onset (y_t - y_t-3 > 30)               stack   8997  62.78  38.79           44.51          50.01
        winter (Nov-Feb)           best_tree  50970  46.87  25.68           31.74          31.94
        winter (Nov-Feb)         climatology  50970  93.15  57.04          -35.66         -51.18
        winter (Nov-Feb)    equal_neural_ens  50970  46.61  25.48           32.11          32.46
        winter (Nov-Feb)         persistence  50970  68.66  37.73            0.00           0.00
        winter (Nov-Feb)               ridge  50970  48.51  27.41           29.35          27.34
        winter (Nov-Feb) same_hour_yesterday  50970  55.41  29.28           19.31          22.41
        winter (Nov-Feb)               stack  50970  46.52  25.34           32.25          32.84
  other months (Mar-Oct)           best_tree  97425  21.53  11.99           36.08          29.79
  other months (Mar-Oct)         climatology  97425  44.59  37.52          -32.41        -119.70
  other months (Mar-Oct)    equal_neural_ens  97425  21.66  11.77           35.70          31.05
  other months (Mar-Oct)         persistence  97425  33.68  17.08            0.00           0.00
  other months (Mar-Oct)               ridge  97425  22.66  12.71           32.73          25.58
  other months (Mar-Oct) same_hour_yesterday  97425  25.58  12.91           24.05          24.42
  other months (Mar-Oct)               stack  97425  21.47  11.73           36.25          31.29
                     all           best_tree 148395  32.54  16.69           33.08          30.95
                     all         climatology 148395  65.47  44.22          -34.64         -82.96
                     all    equal_neural_ens 148395  32.47  16.48           33.22          31.81
                     all         persistence 148395  48.62  24.17            0.00           0.00
                     all               ridge 148395  33.84  17.76           30.40          26.52
                     all same_hour_yesterday 148395  38.52  18.53           20.77          23.34
                     all               stack 148395  32.34  16.41           33.48          32.12
```

### `P2_2023_ALL20_h12_direct_openmeteo`
```text
                  regime               model      N   RMSE    MAE  skill_RMSE_pct  skill_MAE_pct
               y_t >= 60           best_tree  41397  50.82  30.87           38.00          41.16
               y_t >= 60         climatology  41397 102.51  62.06          -25.06         -18.26
               y_t >= 60    equal_neural_ens  41397  49.45  29.96           39.66          42.90
               y_t >= 60         persistence  41397  81.96  52.47            0.00           0.00
               y_t >= 60               ridge  41397  52.85  32.71           35.52          37.67
               y_t >= 60 same_hour_yesterday  41397  61.80  36.13           24.60          31.15
               y_t >= 60               stack  41397  49.14  29.60           40.05          43.59
              y_t >= 120           best_tree  14235  71.10  47.22           41.33          47.70
              y_t >= 120         climatology  14235 157.98 116.13          -30.37         -28.64
              y_t >= 120    equal_neural_ens  14235  68.86  45.58           43.17          49.51
              y_t >= 120         persistence  14235 121.18  90.28            0.00           0.00
              y_t >= 120               ridge  14235  73.82  50.43           39.08          44.14
              y_t >= 120 same_hour_yesterday  14235  85.61  55.32           29.35          38.72
              y_t >= 120               stack  14235  68.39  45.11           43.56          50.03
              y_t >= 250           best_tree   3698  95.64  68.56           45.45          50.06
              y_t >= 250         climatology   3698 222.90 189.59          -27.13         -38.10
              y_t >= 250    equal_neural_ens   3698  90.80  65.48           48.21          52.30
              y_t >= 250         persistence   3698 175.34 137.29            0.00           0.00
              y_t >= 250               ridge   3698  98.06  73.27           44.07          46.63
              y_t >= 250 same_hour_yesterday   3698 113.09  81.79           35.50          40.42
              y_t >= 250               stack   3698  90.43  65.26           48.43          52.46
onset (y_t - y_t-3 > 30)           best_tree   8997  62.27  38.67           44.96          50.17
onset (y_t - y_t-3 > 30)         climatology   8997 128.41  82.31          -13.49          -6.08
onset (y_t - y_t-3 > 30)    equal_neural_ens   8997  60.61  37.39           46.43          51.82
onset (y_t - y_t-3 > 30)         persistence   8997 113.15  77.60            0.00           0.00
onset (y_t - y_t-3 > 30)               ridge   8997  65.72  42.06           41.92          45.80
onset (y_t - y_t-3 > 30) same_hour_yesterday   8997  74.97  45.32           33.74          41.59
onset (y_t - y_t-3 > 30)               stack   8997  60.12  36.86           46.87          52.50
        winter (Nov-Feb)           best_tree  50970  45.46  24.77           33.80          34.36
        winter (Nov-Feb)         climatology  50970  93.15  57.04          -35.66         -51.18
        winter (Nov-Feb)    equal_neural_ens  50970  44.22  24.00           35.59          36.39
        winter (Nov-Feb)         persistence  50970  68.66  37.73            0.00           0.00
        winter (Nov-Feb)               ridge  50970  47.30  26.74           31.11          29.13
        winter (Nov-Feb) same_hour_yesterday  50970  55.41  29.28           19.31          22.41
        winter (Nov-Feb)               stack  50970  43.98  23.73           35.95          37.10
  other months (Mar-Oct)           best_tree  97425  20.84  11.46           38.12          32.92
  other months (Mar-Oct)         climatology  97425  44.59  37.52          -32.41        -119.70
  other months (Mar-Oct)    equal_neural_ens  97425  20.62  11.28           38.78          33.95
  other months (Mar-Oct)         persistence  97425  33.68  17.08            0.00           0.00
  other months (Mar-Oct)               ridge  97425  22.67  13.60           32.70          20.39
  other months (Mar-Oct) same_hour_yesterday  97425  25.58  12.91           24.05          24.42
  other months (Mar-Oct)               stack  97425  20.45  11.03           39.28          35.38
                     all           best_tree 148395  31.54  16.03           35.13          33.69
                     all         climatology 148395  65.47  44.22          -34.64         -82.96
                     all    equal_neural_ens 148395  30.84  15.65           36.58          35.26
                     all         persistence 148395  48.62  24.17            0.00           0.00
                     all               ridge 148395  33.26  18.11           31.60          25.08
                     all same_hour_yesterday 148395  38.52  18.53           20.77          23.34
                     all               stack 148395  30.64  15.40           36.98          36.30
```

### `P2_2023_ALL20_h24_direct_none`
```text
                  regime               model      N   RMSE    MAE  skill_RMSE_pct  skill_MAE_pct
               y_t >= 60           best_tree  41093  57.82  36.06           14.03          11.12
               y_t >= 60         climatology  41093 104.75  64.59          -55.74         -59.20
               y_t >= 60    equal_neural_ens  41093  57.56  35.96           14.43          11.37
               y_t >= 60         persistence  41093  67.26  40.57            0.00           0.00
               y_t >= 60               ridge  41093  59.70  36.95           11.25           8.92
               y_t >= 60 same_hour_yesterday  41093  67.26  40.57            0.00           0.00
               y_t >= 60               stack  41093  57.93  35.92           13.88          11.46
              y_t >= 120           best_tree  14148  84.85  59.70           16.92          15.65
              y_t >= 120         climatology  14148 167.32 129.58          -63.83         -83.11
              y_t >= 120    equal_neural_ens  14148  84.51  59.59           17.25          15.80
              y_t >= 120         persistence  14148 102.13  70.77            0.00           0.00
              y_t >= 120               ridge  14148  88.27  62.04           13.57          12.33
              y_t >= 120 same_hour_yesterday  14148 102.13  70.77            0.00           0.00
              y_t >= 120               stack  14148  85.11  59.54           16.66          15.87
              y_t >= 250           best_tree   3646 108.61  81.91           27.45          26.69
              y_t >= 250         climatology   3646 247.77 217.34          -65.50         -94.51
              y_t >= 250    equal_neural_ens   3646 108.13  81.82           27.77          26.77
              y_t >= 250         persistence   3646 149.70 111.74            0.00           0.00
              y_t >= 250               ridge   3646 115.95  88.78           22.55          20.54
              y_t >= 250 same_hour_yesterday   3646 149.70 111.74            0.00           0.00
              y_t >= 250               stack   3646 108.78  81.88           27.33          26.72
onset (y_t - y_t-3 > 30)           best_tree   8966  74.82  49.03           23.09          21.63
onset (y_t - y_t-3 > 30)         climatology   8966 142.35  94.70          -46.33         -51.36
onset (y_t - y_t-3 > 30)    equal_neural_ens   8966  74.65  49.09           23.27          21.53
onset (y_t - y_t-3 > 30)         persistence   8966  97.28  62.56            0.00           0.00
onset (y_t - y_t-3 > 30)               ridge   8966  78.85  51.59           18.94          17.54
onset (y_t - y_t-3 > 30) same_hour_yesterday   8966  97.28  62.56            0.00           0.00
onset (y_t - y_t-3 > 30)               stack   8966  75.06  48.85           22.84          21.91
        winter (Nov-Feb)           best_tree  50516  49.22  27.53           11.91           6.37
        winter (Nov-Feb)         climatology  50516  93.05  57.01          -66.54         -93.89
        winter (Nov-Feb)    equal_neural_ens  50516  48.92  27.45           12.43           6.65
        winter (Nov-Feb)         persistence  50516  55.87  29.40            0.00           0.00
        winter (Nov-Feb)               ridge  50516  50.74  29.14            9.18           0.88
        winter (Nov-Feb) same_hour_yesterday  50516  55.87  29.40            0.00           0.00
        winter (Nov-Feb)               stack  50516  49.26  27.25           11.84           7.33
  other months (Mar-Oct)           best_tree  96235  21.99  12.65           14.48           2.36
  other months (Mar-Oct)         climatology  96235  44.50  37.43          -73.02        -188.98
  other months (Mar-Oct)    equal_neural_ens  96235  21.92  12.12           14.78           6.42
  other months (Mar-Oct)         persistence  96235  25.72  12.95            0.00           0.00
  other months (Mar-Oct)               ridge  96235  22.71  12.59           11.69           2.78
  other months (Mar-Oct) same_hour_yesterday  96235  25.72  12.95            0.00           0.00
  other months (Mar-Oct)               stack  96235  21.96  12.11           14.60           6.50
                     all           best_tree 146751  33.93  17.77           12.64           4.54
                     all         climatology 146751  65.41  44.17          -68.43        -137.28
                     all    equal_neural_ens 146751  33.75  17.40           13.10           6.55
                     all         persistence 146751  38.84  18.61            0.00           0.00
                     all               ridge 146751  34.99  18.29            9.90           1.75
                     all same_hour_yesterday 146751  38.84  18.61            0.00           0.00
                     all               stack 146751  33.93  17.32           12.62           6.95
```

### `P2_2023_ALL20_h24_direct_openmeteo`
```text
                  regime               model      N   RMSE    MAE  skill_RMSE_pct  skill_MAE_pct
               y_t >= 60           best_tree  41093  56.62  35.05           15.82          13.60
               y_t >= 60         climatology  41093 104.75  64.59          -55.74         -59.20
               y_t >= 60    equal_neural_ens  41093  55.06  34.27           18.14          15.53
               y_t >= 60         persistence  41093  67.26  40.57            0.00           0.00
               y_t >= 60               ridge  41093  58.00  36.36           13.77          10.38
               y_t >= 60 same_hour_yesterday  41093  67.26  40.57            0.00           0.00
               y_t >= 60               stack  41093  55.53  34.17           17.44          15.78
              y_t >= 120           best_tree  14148  83.39  57.87           18.35          18.22
              y_t >= 120         climatology  14148 167.32 129.58          -63.83         -83.11
              y_t >= 120    equal_neural_ens  14148  80.77  56.13           20.91          20.68
              y_t >= 120         persistence  14148 102.13  70.77            0.00           0.00
              y_t >= 120               ridge  14148  85.93  60.45           15.86          14.58
              y_t >= 120 same_hour_yesterday  14148 102.13  70.77            0.00           0.00
              y_t >= 120               stack  14148  81.64  56.20           20.06          20.58
              y_t >= 250           best_tree   3646 110.49  82.45           26.20          26.21
              y_t >= 250         climatology   3646 247.77 217.34          -65.50         -94.51
              y_t >= 250    equal_neural_ens   3646 105.10  79.10           29.80          29.21
              y_t >= 250         persistence   3646 149.70 111.74            0.00           0.00
              y_t >= 250               ridge   3646 113.24  86.81           24.36          22.31
              y_t >= 250 same_hour_yesterday   3646 149.70 111.74            0.00           0.00
              y_t >= 250               stack   3646 106.18  79.69           29.07          28.68
onset (y_t - y_t-3 > 30)           best_tree   8966  74.13  48.06           23.80          23.18
onset (y_t - y_t-3 > 30)         climatology   8966 142.35  94.70          -46.33         -51.36
onset (y_t - y_t-3 > 30)    equal_neural_ens   8966  71.19  46.01           26.82          26.46
onset (y_t - y_t-3 > 30)         persistence   8966  97.28  62.56            0.00           0.00
onset (y_t - y_t-3 > 30)               ridge   8966  76.38  50.22           21.48          19.72
onset (y_t - y_t-3 > 30) same_hour_yesterday   8966  97.28  62.56            0.00           0.00
onset (y_t - y_t-3 > 30)               stack   8966  72.05  46.10           25.94          26.32
        winter (Nov-Feb)           best_tree  50516  48.21  26.60           13.71           9.51
        winter (Nov-Feb)         climatology  50516  93.05  57.01          -66.54         -93.89
        winter (Nov-Feb)    equal_neural_ens  50516  46.78  25.99           16.27          11.60
        winter (Nov-Feb)         persistence  50516  55.87  29.40            0.00           0.00
        winter (Nov-Feb)               ridge  50516  49.30  28.03           11.76           4.65
        winter (Nov-Feb) same_hour_yesterday  50516  55.87  29.40            0.00           0.00
        winter (Nov-Feb)               stack  50516  47.20  25.71           15.51          12.56
  other months (Mar-Oct)           best_tree  96235  21.71  12.65           15.61           2.30
  other months (Mar-Oct)         climatology  96235  44.50  37.43          -73.02        -188.98
  other months (Mar-Oct)    equal_neural_ens  96235  21.31  11.77           17.13           9.12
  other months (Mar-Oct)         persistence  96235  25.72  12.95            0.00           0.00
  other months (Mar-Oct)               ridge  96235  23.03  13.97           10.45          -7.89
  other months (Mar-Oct) same_hour_yesterday  96235  25.72  12.95            0.00           0.00
  other months (Mar-Oct)               stack  96235  21.27  11.60           17.30          10.47
                     all           best_tree 146751  33.30  17.46           14.25           6.22
                     all         climatology 146751  65.41  44.17          -68.43        -137.28
                     all    equal_neural_ens 146751  32.42  16.67           16.52          10.47
                     all         persistence 146751  38.84  18.61            0.00           0.00
                     all               ridge 146751  34.42  18.81           11.38          -1.07
                     all same_hour_yesterday 146751  38.84  18.61            0.00           0.00
                     all               stack 146751  32.61  16.45           16.02          11.61
```

### `P2_2024_ALL19_h1_direct_none`
```text
                  regime            model      N   RMSE    MAE  skill_RMSE_pct  skill_MAE_pct
               y_t >= 60        best_tree  32782  26.59  15.47           11.52          12.00
               y_t >= 60      climatology  32782 119.93  73.41         -299.06        -317.65
               y_t >= 60 equal_neural_ens  32782  26.24  15.22           12.67          13.41
               y_t >= 60      persistence  32782  30.05  17.58            0.00           0.00
               y_t >= 60            ridge  32782  27.66  16.12            7.96           8.31
               y_t >= 60            stack  32782  26.24  15.22           12.70          13.40
              y_t >= 120        best_tree  12375  37.61  24.53           12.07          12.42
              y_t >= 120      climatology  12375 191.06 156.47         -346.65        -458.55
              y_t >= 120 equal_neural_ens  12375  37.06  24.11           13.37          13.93
              y_t >= 120      persistence  12375  42.78  28.01            0.00           0.00
              y_t >= 120            ridge  12375  39.17  25.60            8.44           8.61
              y_t >= 120            stack  12375  37.04  24.10           13.42          13.98
              y_t >= 250        best_tree   3570  53.17  37.47           12.18          10.64
              y_t >= 250      climatology   3570 304.74 281.00         -403.37        -570.08
              y_t >= 250 equal_neural_ens   3570  52.06  36.63           14.00          12.65
              y_t >= 250      persistence   3570  60.54  41.94            0.00           0.00
              y_t >= 250            ridge   3570  55.37  38.63            8.54           7.88
              y_t >= 250            stack   3570  52.02  36.59           14.08          12.75
onset (y_t - y_t-3 > 30)        best_tree   7911  40.08  25.86           12.01          12.09
onset (y_t - y_t-3 > 30)      climatology   7911 177.38 121.97         -289.40        -314.55
onset (y_t - y_t-3 > 30) equal_neural_ens   7911  39.55  25.61           13.16          12.95
onset (y_t - y_t-3 > 30)      persistence   7911  45.55  29.42            0.00           0.00
onset (y_t - y_t-3 > 30)            ridge   7911  42.00  27.12            7.79           7.83
onset (y_t - y_t-3 > 30)            stack   7911  39.52  25.61           13.23          12.95
        winter (Nov-Feb)        best_tree  51024  17.88   8.64           11.56          10.47
        winter (Nov-Feb)      climatology  51024  94.07  54.20         -365.35        -461.85
        winter (Nov-Feb) equal_neural_ens  51024  17.54   8.47           13.22          12.22
        winter (Nov-Feb)      persistence  51024  20.22   9.65            0.00           0.00
        winter (Nov-Feb)            ridge  51024  18.58   9.08            8.11           5.91
        winter (Nov-Feb)            stack  51024  17.53   8.47           13.29          12.16
  other months (Mar-Oct)        best_tree  95645  11.10   5.14           10.06           7.30
  other months (Mar-Oct)      climatology  95645  45.60  39.11         -269.34        -606.02
  other months (Mar-Oct) equal_neural_ens  95645  11.12   5.13            9.90           7.43
  other months (Mar-Oct)      persistence  95645  12.35   5.54            0.00           0.00
  other months (Mar-Oct)            ridge  95645  11.64   5.42            5.71           2.10
  other months (Mar-Oct)            stack  95645  11.12   5.14            9.92           7.31
                     all        best_tree 146669  13.84   6.35           10.94           8.82
                     all      climatology 146669  66.59  44.36         -328.46        -536.59
                     all equal_neural_ens 146669  13.70   6.29           11.84           9.74
                     all      persistence 146669  15.54   6.97            0.00           0.00
                     all            ridge 146669  14.44   6.69            7.11           3.94
                     all            stack 146669  13.69   6.30           11.89           9.64
```

### `P2_2024_ALL19_h1_direct_openmeteo`
```text
                  regime            model      N   RMSE    MAE  skill_RMSE_pct  skill_MAE_pct
               y_t >= 60        best_tree  32782  26.68  15.49           11.24          11.86
               y_t >= 60      climatology  32782 119.93  73.41         -299.06        -317.65
               y_t >= 60 equal_neural_ens  32782  26.16  15.30           12.95          12.95
               y_t >= 60      persistence  32782  30.05  17.58            0.00           0.00
               y_t >= 60            ridge  32782  27.43  16.01            8.73           8.89
               y_t >= 60            stack  32782  26.21  15.31           12.78          12.89
              y_t >= 120        best_tree  12375  37.74  24.60           11.77          12.18
              y_t >= 120      climatology  12375 191.06 156.47         -346.65        -458.55
              y_t >= 120 equal_neural_ens  12375  36.75  24.07           14.08          14.07
              y_t >= 120      persistence  12375  42.78  28.01            0.00           0.00
              y_t >= 120            ridge  12375  38.86  25.33            9.16           9.60
              y_t >= 120            stack  12375  36.83  24.09           13.89          14.00
              y_t >= 250        best_tree   3570  53.29  37.53           11.98          10.51
              y_t >= 250      climatology   3570 304.74 281.00         -403.37        -570.08
              y_t >= 250 equal_neural_ens   3570  51.25  36.14           15.34          13.81
              y_t >= 250      persistence   3570  60.54  41.94            0.00           0.00
              y_t >= 250            ridge   3570  55.05  38.30            9.07           8.67
              y_t >= 250            stack   3570  51.43  36.28           15.05          13.50
onset (y_t - y_t-3 > 30)        best_tree   7911  40.34  25.88           11.43          12.03
onset (y_t - y_t-3 > 30)      climatology   7911 177.38 121.97         -289.40        -314.55
onset (y_t - y_t-3 > 30) equal_neural_ens   7911  39.36  25.54           13.59          13.20
onset (y_t - y_t-3 > 30)      persistence   7911  45.55  29.42            0.00           0.00
onset (y_t - y_t-3 > 30)            ridge   7911  41.76  26.85            8.32           8.73
onset (y_t - y_t-3 > 30)            stack   7911  39.47  25.62           13.35          12.92
        winter (Nov-Feb)        best_tree  51024  17.93   8.67           11.29          10.18
        winter (Nov-Feb)      climatology  51024  94.07  54.20         -365.35        -461.85
        winter (Nov-Feb) equal_neural_ens  51024  17.28   8.48           14.54          12.09
        winter (Nov-Feb)      persistence  51024  20.22   9.65            0.00           0.00
        winter (Nov-Feb)            ridge  51024  18.45   9.17            8.72           4.91
        winter (Nov-Feb)            stack  51024  17.32   8.51           14.30          11.84
  other months (Mar-Oct)        best_tree  95645  11.13   5.13            9.86           7.35
  other months (Mar-Oct)      climatology  95645  45.60  39.11         -269.34        -606.02
  other months (Mar-Oct) equal_neural_ens  95645  11.29   5.20            8.56           6.10
  other months (Mar-Oct)      persistence  95645  12.35   5.54            0.00           0.00
  other months (Mar-Oct)            ridge  95645  11.64   5.62            5.73          -1.50
  other months (Mar-Oct)            stack  95645  11.28   5.21            8.60           5.98
                     all        best_tree 146669  13.88   6.36           10.70           8.72
                     all      climatology 146669  66.59  44.36         -328.46        -536.59
                     all equal_neural_ens 146669  13.67   6.34           12.03           8.98
                     all      persistence 146669  15.54   6.97            0.00           0.00
                     all            ridge 146669  14.38   6.86            7.48           1.59
                     all            stack 146669  13.69   6.36           11.91           8.80
```

### `P2_2024_ALL19_h24_direct_none`
```text
                  regime               model      N   RMSE    MAE  skill_RMSE_pct  skill_MAE_pct
               y_t >= 60           best_tree  31980  65.21  39.93           13.46          12.64
               y_t >= 60         climatology  31980 120.94  74.00          -60.51         -61.87
               y_t >= 60    equal_neural_ens  31980  64.94  40.23           13.82          11.99
               y_t >= 60         persistence  31980  75.35  45.71            0.00           0.00
               y_t >= 60               ridge  31980  67.36  40.65           10.61          11.09
               y_t >= 60 same_hour_yesterday  31980  75.35  45.71            0.00           0.00
               y_t >= 60               stack  31980  64.95  40.11           13.80          12.25
              y_t >= 120           best_tree  12128  93.90  63.62           15.26          16.78
              y_t >= 120         climatology  12128 186.65 144.34          -68.45         -88.79
              y_t >= 120    equal_neural_ens  12128  93.43  64.18           15.68          16.05
              y_t >= 120         persistence  12128 110.80  76.45            0.00           0.00
              y_t >= 120               ridge  12128  97.68  65.54           11.84          14.27
              y_t >= 120 same_hour_yesterday  12128 110.80  76.45            0.00           0.00
              y_t >= 120               stack  12128  93.50  63.96           15.62          16.35
              y_t >= 250           best_tree   3522 126.19  90.56           22.29          26.62
              y_t >= 250         climatology   3522 267.74 226.77          -64.88         -83.75
              y_t >= 250    equal_neural_ens   3522 125.64  91.47           22.63          25.88
              y_t >= 250         persistence   3522 162.39 123.41            0.00           0.00
              y_t >= 250               ridge   3522 133.86  97.23           17.57          21.21
              y_t >= 250 same_hour_yesterday   3522 162.39 123.41            0.00           0.00
              y_t >= 250               stack   3522 125.87  91.29           22.49          26.03
onset (y_t - y_t-3 > 30)           best_tree   7636  79.16  50.16           20.45          22.52
onset (y_t - y_t-3 > 30)         climatology   7636 155.35 101.49          -56.12         -56.76
onset (y_t - y_t-3 > 30)    equal_neural_ens   7636  78.51  50.33           21.10          22.26
onset (y_t - y_t-3 > 30)         persistence   7636  99.50  64.74            0.00           0.00
onset (y_t - y_t-3 > 30)               ridge   7636  81.94  51.69           17.65          20.17
onset (y_t - y_t-3 > 30) same_hour_yesterday   7636  99.50  64.74            0.00           0.00
onset (y_t - y_t-3 > 30)               stack   7636  78.59  50.18           21.02          22.49
        winter (Nov-Feb)           best_tree  50079  49.12  25.32           13.34           9.05
        winter (Nov-Feb)         climatology  50079  95.14  54.76          -67.87         -96.71
        winter (Nov-Feb)    equal_neural_ens  50079  48.94  25.61           13.65           8.01
        winter (Nov-Feb)         persistence  50079  56.68  27.84            0.00           0.00
        winter (Nov-Feb)               ridge  50079  51.01  27.11            9.99           2.60
        winter (Nov-Feb) same_hour_yesterday  50079  56.68  27.84            0.00           0.00
        winter (Nov-Feb)               stack  50079  48.95  25.51           13.64           8.36
  other months (Mar-Oct)           best_tree  92803  20.42  11.09            9.52           4.17
  other months (Mar-Oct)         climatology  92803  45.68  39.20         -102.42        -238.73
  other months (Mar-Oct)    equal_neural_ens  92803  20.32  10.90            9.97           5.81
  other months (Mar-Oct)         persistence  92803  22.57  11.57            0.00           0.00
  other months (Mar-Oct)               ridge  92803  21.02  11.69            6.84          -1.06
  other months (Mar-Oct) same_hour_yesterday  92803  22.57  11.57            0.00           0.00
  other months (Mar-Oct)               stack  92803  20.30  10.92           10.04           5.66
                     all           best_tree 142882  33.41  16.08           12.45           6.92
                     all         climatology 142882  67.29  44.65          -76.31        -158.51
                     all    equal_neural_ens 142882  33.28  16.05           12.80           7.05
                     all         persistence 142882  38.17  17.27            0.00           0.00
                     all               ridge 142882  34.63  17.10            9.27           1.01
                     all same_hour_yesterday 142882  38.17  17.27            0.00           0.00
                     all               stack 142882  33.28  16.03           12.81           7.19
```

### `P2_2024_ALL19_h24_direct_openmeteo`
```text
                  regime               model      N   RMSE    MAE  skill_RMSE_pct  skill_MAE_pct
               y_t >= 60           best_tree  31980  63.48  39.32           15.76          13.99
               y_t >= 60         climatology  31980 120.94  74.00          -60.51         -61.87
               y_t >= 60    equal_neural_ens  31980  63.18  38.56           16.15          15.65
               y_t >= 60         persistence  31980  75.35  45.71            0.00           0.00
               y_t >= 60               ridge  31980  66.27  40.71           12.06          10.95
               y_t >= 60 same_hour_yesterday  31980  75.35  45.71            0.00           0.00
               y_t >= 60               stack  31980  62.55  38.41           16.99          15.97
              y_t >= 120           best_tree  12128  91.89  63.64           17.07          16.76
              y_t >= 120         climatology  12128 186.65 144.34          -68.45         -88.79
              y_t >= 120    equal_neural_ens  12128  91.63  61.91           17.30          19.03
              y_t >= 120         persistence  12128 110.80  76.45            0.00           0.00
              y_t >= 120               ridge  12128  96.19  64.72           13.19          15.35
              y_t >= 120 same_hour_yesterday  12128 110.80  76.45            0.00           0.00
              y_t >= 120               stack  12128  90.55  61.78           18.28          19.19
              y_t >= 250           best_tree   3522 122.87  90.25           24.33          26.87
              y_t >= 250         climatology   3522 267.74 226.77          -64.88         -83.75
              y_t >= 250    equal_neural_ens   3522 123.97  89.40           23.66          27.56
              y_t >= 250         persistence   3522 162.39 123.41            0.00           0.00
              y_t >= 250               ridge   3522 131.87  95.02           18.79          23.01
              y_t >= 250 same_hour_yesterday   3522 162.39 123.41            0.00           0.00
              y_t >= 250               stack   3522 122.28  89.06           24.70          27.83
onset (y_t - y_t-3 > 30)           best_tree   7636  76.83  49.46           22.79          23.61
onset (y_t - y_t-3 > 30)         climatology   7636 155.35 101.49          -56.12         -56.76
onset (y_t - y_t-3 > 30)    equal_neural_ens   7636  76.80  48.05           22.81          25.78
onset (y_t - y_t-3 > 30)         persistence   7636  99.50  64.74            0.00           0.00
onset (y_t - y_t-3 > 30)               ridge   7636  80.78  51.52           18.81          20.43
onset (y_t - y_t-3 > 30) same_hour_yesterday   7636  99.50  64.74            0.00           0.00
onset (y_t - y_t-3 > 30)               stack   7636  75.98  47.95           23.64          25.94
        winter (Nov-Feb)           best_tree  50079  47.79  25.16           15.68           9.59
        winter (Nov-Feb)         climatology  50079  95.14  54.76          -67.87         -96.71
        winter (Nov-Feb)    equal_neural_ens  50079  47.65  25.04           15.92          10.05
        winter (Nov-Feb)         persistence  50079  56.68  27.84            0.00           0.00
        winter (Nov-Feb)               ridge  50079  50.32  27.09           11.22           2.69
        winter (Nov-Feb) same_hour_yesterday  50079  56.68  27.84            0.00           0.00
        winter (Nov-Feb)               stack  50079  47.11  24.80           16.88          10.91
  other months (Mar-Oct)           best_tree  92803  20.12  10.87           10.85           6.07
  other months (Mar-Oct)         climatology  92803  45.68  39.20         -102.42        -238.73
  other months (Mar-Oct)    equal_neural_ens  92803  20.15  11.02           10.71           4.78
  other months (Mar-Oct)         persistence  92803  22.57  11.57            0.00           0.00
  other months (Mar-Oct)               ridge  92803  21.83  13.12            3.26         -13.38
  other months (Mar-Oct) same_hour_yesterday  92803  22.57  11.57            0.00           0.00
  other months (Mar-Oct)               stack  92803  20.02  10.90           11.28           5.78
                     all           best_tree 142882  32.61  15.88           14.56           8.06
                     all         climatology 142882  67.29  44.65          -76.31        -158.51
                     all    equal_neural_ens 142882  32.55  15.93           14.71           7.76
                     all         persistence 142882  38.17  17.27            0.00           0.00
                     all               ridge 142882  34.60  18.02            9.35          -4.31
                     all same_hour_yesterday 142882  38.17  17.27            0.00           0.00
                     all               stack 142882  32.22  15.77           15.57           8.68
```

### `P1_LOCO_Delhi_h1_direct_none`
```text
                  regime            model      N   RMSE    MAE  skill_RMSE_pct  skill_MAE_pct
               y_t >= 60        best_tree 126201  35.26  21.98            6.51           8.08
               y_t >= 60      climatology 126201 162.60 120.04         -331.07        -401.93
               y_t >= 60 equal_neural_ens 126201  33.91  21.05           10.10          11.98
               y_t >= 60      persistence 126201  37.72  23.92            0.00           0.00
               y_t >= 60            ridge 126201  35.59  22.78            5.64           4.76
               y_t >= 60            stack 126201  34.26  21.15            9.17          11.58
              y_t >= 120        best_tree  68674  44.06  28.60            6.15           7.46
              y_t >= 120      climatology  68674 215.26 183.33         -358.55        -493.18
              y_t >= 120 equal_neural_ens  68674  42.23  27.14           10.05          12.18
              y_t >= 120      persistence  68674  46.95  30.91            0.00           0.00
              y_t >= 120            ridge  68674  44.22  29.34            5.81           5.07
              y_t >= 120            stack  68674  42.75  27.32            8.94          11.60
              y_t >= 250        best_tree  21079  63.40  43.09            3.33           1.63
              y_t >= 250      climatology  21079 328.22 305.58         -400.42        -597.57
              y_t >= 250 equal_neural_ens  21079  60.51  40.31            7.74           7.97
              y_t >= 250      persistence  21079  65.59  43.81            0.00           0.00
              y_t >= 250            ridge  21079  61.85  42.25            5.69           3.56
              y_t >= 250            stack  21079  61.61  40.80            6.06           6.85
onset (y_t - y_t-3 > 30)        best_tree  39708  48.65  30.98            5.01           6.41
onset (y_t - y_t-3 > 30)      climatology  39708 214.34 165.78         -318.51        -400.79
onset (y_t - y_t-3 > 30) equal_neural_ens  39708  47.00  29.72            8.23          10.23
onset (y_t - y_t-3 > 30)      persistence  39708  51.22  33.10            0.00           0.00
onset (y_t - y_t-3 > 30)            ridge  39708  48.92  31.73            4.48           4.16
onset (y_t - y_t-3 > 30)            stack  39708  47.55  29.89            7.16           9.70
        winter (Nov-Feb)        best_tree  71852  37.34  23.81            6.66           7.81
        winter (Nov-Feb)      climatology  71852 196.47 153.52         -391.11        -494.49
        winter (Nov-Feb) equal_neural_ens  71852  35.81  22.44           10.48          13.11
        winter (Nov-Feb)      persistence  71852  40.01  25.82            0.00           0.00
        winter (Nov-Feb)            ridge  71852  38.07  25.10            4.84           2.82
        winter (Nov-Feb)            stack  71852  36.34  22.60            9.16          12.49
  other months (Mar-Oct)        best_tree 136196  22.76  12.77            6.10           6.77
  other months (Mar-Oct)      climatology 136196  66.36  40.29         -173.76        -194.13
  other months (Mar-Oct) equal_neural_ens 136196  22.05  12.54            9.01           8.42
  other months (Mar-Oct)      persistence 136196  24.24  13.70            0.00           0.00
  other months (Mar-Oct)            ridge 136196  22.71  13.04            6.31           4.80
  other months (Mar-Oct)            stack 136196  22.10  12.56            8.84           8.32
                     all        best_tree 208048  28.65  16.58            6.43           7.29
                     all      climatology 208048 127.33  79.39         -315.90        -343.90
                     all equal_neural_ens 208048  27.59  15.96            9.87          10.76
                     all      persistence 208048  30.62  17.88            0.00           0.00
                     all            ridge 208048  28.95  17.20            5.44           3.81
                     all            stack 208048  27.85  16.03            9.03          10.39
```

### `P1_LOCO_Delhi_h1_direct_openmeteo`
```text
                  regime            model      N   RMSE    MAE  skill_RMSE_pct  skill_MAE_pct
               y_t >= 60        best_tree 126201  36.54  22.97            3.13           3.94
               y_t >= 60      climatology 126201 162.60 120.04         -331.07        -401.93
               y_t >= 60 equal_neural_ens 126201  34.69  21.14            8.04          11.60
               y_t >= 60      persistence 126201  37.72  23.92            0.00           0.00
               y_t >= 60            ridge 126201  35.36  22.50            6.27           5.93
               y_t >= 60            stack 126201  33.91  20.78           10.10          13.10
              y_t >= 120        best_tree  68674  45.99  30.50            2.03           1.32
              y_t >= 120      climatology  68674 215.26 183.33         -358.55        -493.18
              y_t >= 120 equal_neural_ens  68674  43.46  27.47            7.42          11.11
              y_t >= 120      persistence  68674  46.95  30.91            0.00           0.00
              y_t >= 120            ridge  68674  44.02  29.07            6.23           5.94
              y_t >= 120            stack  68674  42.32  26.81            9.84          13.26
              y_t >= 250        best_tree  21079  67.64  48.94           -3.12         -11.72
              y_t >= 250      climatology  21079 328.22 305.58         -400.42        -597.57
              y_t >= 250 equal_neural_ens  21079  63.64  42.00            2.98           4.11
              y_t >= 250      persistence  21079  65.59  43.81            0.00           0.00
              y_t >= 250            ridge  21079  61.95  42.31            5.54           3.43
              y_t >= 250            stack  21079  61.16  39.94            6.75           8.84
onset (y_t - y_t-3 > 30)        best_tree  39708  50.54  32.78            1.32           0.97
onset (y_t - y_t-3 > 30)      climatology  39708 214.34 165.78         -318.51        -400.79
onset (y_t - y_t-3 > 30) equal_neural_ens  39708  48.53  30.13            5.25           8.99
onset (y_t - y_t-3 > 30)      persistence  39708  51.22  33.10            0.00           0.00
onset (y_t - y_t-3 > 30)            ridge  39708  48.82  31.49            4.69           4.88
onset (y_t - y_t-3 > 30)            stack  39708  47.29  29.41            7.66          11.16
        winter (Nov-Feb)        best_tree  71852  39.36  25.43            1.62           1.51
        winter (Nov-Feb)      climatology  71852 196.47 153.52         -391.11        -494.49
        winter (Nov-Feb) equal_neural_ens  71852  36.93  22.59            7.69          12.52
        winter (Nov-Feb)      persistence  71852  40.01  25.82            0.00           0.00
        winter (Nov-Feb)            ridge  71852  37.87  24.85            5.34           3.75
        winter (Nov-Feb)            stack  71852  35.83  22.08           10.44          14.50
  other months (Mar-Oct)        best_tree 136196  22.78  12.79            6.02           6.64
  other months (Mar-Oct)      climatology 136196  66.36  40.29         -173.76        -194.13
  other months (Mar-Oct) equal_neural_ens 136196  22.18  12.50            8.49           8.71
  other months (Mar-Oct)      persistence 136196  24.24  13.70            0.00           0.00
  other months (Mar-Oct)            ridge 136196  22.50  12.90            7.18           5.80
  other months (Mar-Oct)            stack 136196  22.01  12.48            9.18           8.92
                     all        best_tree 208048  29.57  17.15            3.40           4.08
                     all      climatology 208048 127.33  79.39         -315.90        -343.90
                     all equal_neural_ens 208048  28.16  15.99            8.02          10.61
                     all      persistence 208048  30.62  17.88            0.00           0.00
                     all            ridge 208048  28.75  17.03            6.09           4.78
                     all            stack 208048  27.58  15.79            9.93          11.70
```

### `P1_LOCO_Delhi_h24_direct_none`
```text
                  regime               model      N   RMSE    MAE  skill_RMSE_pct  skill_MAE_pct
               y_t >= 60           best_tree 122464  89.76  56.93            2.82           4.99
               y_t >= 60         climatology 122464 161.04 115.78          -74.35         -93.22
               y_t >= 60    equal_neural_ens 122464  90.49  57.46            2.04           4.11
               y_t >= 60         persistence 122464  92.37  59.92            0.00           0.00
               y_t >= 60               ridge 122464  85.48  56.11            7.45           6.35
               y_t >= 60 same_hour_yesterday 122464  92.37  59.92            0.00           0.00
               y_t >= 60               stack 122464  90.49  57.66            2.04           3.77
              y_t >= 120           best_tree  66502 112.19  76.43            3.23           5.95
              y_t >= 120         climatology  66502 207.68 168.11          -79.15        -106.87
              y_t >= 120    equal_neural_ens  66502 113.19  77.39            2.36           4.77
              y_t >= 120         persistence  66502 115.93  81.26            0.00           0.00
              y_t >= 120               ridge  66502 105.95  74.47            8.61           8.36
              y_t >= 120 same_hour_yesterday  66502 115.93  81.26            0.00           0.00
              y_t >= 120               stack  66502 113.15  77.70            2.40           4.39
              y_t >= 250           best_tree  20451 151.99 109.85            7.25          11.15
              y_t >= 250         climatology  20451 288.78 251.04          -76.23        -103.05
              y_t >= 250    equal_neural_ens  20451 152.94 111.02            6.67          10.20
              y_t >= 250         persistence  20451 163.87 123.63            0.00           0.00
              y_t >= 250               ridge  20451 137.42 102.16           16.14          17.37
              y_t >= 250 same_hour_yesterday  20451 163.87 123.63            0.00           0.00
              y_t >= 250               stack  20451 151.95 110.63            7.27          10.51
onset (y_t - y_t-3 > 30)           best_tree  38518 102.61  65.74            6.52           9.18
onset (y_t - y_t-3 > 30)         climatology  38518 186.19 136.76          -69.63         -88.93
onset (y_t - y_t-3 > 30)    equal_neural_ens  38518 102.96  66.20            6.20           8.55
onset (y_t - y_t-3 > 30)         persistence  38518 109.76  72.39            0.00           0.00
onset (y_t - y_t-3 > 30)               ridge  38518  94.97  63.11           13.48          12.81
onset (y_t - y_t-3 > 30) same_hour_yesterday  38518 109.76  72.39            0.00           0.00
onset (y_t - y_t-3 > 30)               stack  38518 103.10  66.46            6.07           8.18
        winter (Nov-Feb)           best_tree  69736 107.56  70.64            0.97           2.43
        winter (Nov-Feb)         climatology  69736 197.20 153.78          -81.57        -112.41
        winter (Nov-Feb)    equal_neural_ens  69736 107.77  70.77            0.77           2.25
        winter (Nov-Feb)         persistence  69736 108.61  72.40            0.00           0.00
        winter (Nov-Feb)               ridge  69736 101.98  69.38            6.10           4.17
        winter (Nov-Feb) same_hour_yesterday  69736 108.61  72.40            0.00           0.00
        winter (Nov-Feb)               stack  69736 107.48  70.93            1.03           2.02
  other months (Mar-Oct)           best_tree 131913  45.48  26.92            8.81           8.20
  other months (Mar-Oct)         climatology 131913  66.07  40.13          -32.48         -36.84
  other months (Mar-Oct)    equal_neural_ens 131913  46.60  27.28            6.56           6.99
  other months (Mar-Oct)         persistence 131913  49.87  29.32            0.00           0.00
  other months (Mar-Oct)               ridge 131913  44.44  26.85           10.90           8.45
  other months (Mar-Oct) same_hour_yesterday 131913  49.87  29.32            0.00           0.00
  other months (Mar-Oct)               stack 131913  46.91  27.43            5.94           6.45
                     all           best_tree 201649  73.17  42.04            3.14           4.93
                     all         climatology 201649 127.69  79.43          -69.03         -79.63
                     all    equal_neural_ens 201649  73.74  42.32            2.39           4.30
                     all         persistence 201649  75.54  44.22            0.00           0.00
                     all               ridge 201649  69.92  41.55            7.44           6.03
                     all same_hour_yesterday 201649  75.54  44.22            0.00           0.00
                     all               stack 201649  73.72  42.48            2.41           3.94
```

### `P1_LOCO_Delhi_h24_direct_openmeteo`
```text
                  regime               model      N   RMSE    MAE  skill_RMSE_pct  skill_MAE_pct
               y_t >= 60           best_tree 122464  92.62  58.25           -0.27           2.78
               y_t >= 60         climatology 122464 161.04 115.78          -74.35         -93.22
               y_t >= 60    equal_neural_ens 122464  87.72  55.27            5.03           7.76
               y_t >= 60         persistence 122464  92.37  59.92            0.00           0.00
               y_t >= 60               ridge 122464  83.67  54.64            9.42           8.82
               y_t >= 60 same_hour_yesterday 122464  92.37  59.92            0.00           0.00
               y_t >= 60               stack 122464  89.51  56.33            3.09           6.00
              y_t >= 120           best_tree  66502 116.27  78.97           -0.29           2.82
              y_t >= 120         climatology  66502 207.68 168.11          -79.15        -106.87
              y_t >= 120    equal_neural_ens  66502 109.64  74.19            5.42           8.70
              y_t >= 120         persistence  66502 115.93  81.26            0.00           0.00
              y_t >= 120               ridge  66502 103.89  72.60           10.38          10.66
              y_t >= 120 same_hour_yesterday  66502 115.93  81.26            0.00           0.00
              y_t >= 120               stack  66502 111.89  75.79            3.49           6.73
              y_t >= 250           best_tree  20451 159.99 115.84            2.37           6.31
              y_t >= 250         climatology  20451 288.78 251.04          -76.23        -103.05
              y_t >= 250    equal_neural_ens  20451 149.33 107.41            8.87          13.12
              y_t >= 250         persistence  20451 163.87 123.63            0.00           0.00
              y_t >= 250               ridge  20451 135.74 100.60           17.17          18.63
              y_t >= 250 same_hour_yesterday  20451 163.87 123.63            0.00           0.00
              y_t >= 250               stack  20451 151.92 109.64            7.29          11.32
onset (y_t - y_t-3 > 30)           best_tree  38518 106.02  67.39            3.41           6.91
onset (y_t - y_t-3 > 30)         climatology  38518 186.19 136.76          -69.63         -88.93
onset (y_t - y_t-3 > 30)    equal_neural_ens  38518 100.66  64.11            8.30          11.43
onset (y_t - y_t-3 > 30)         persistence  38518 109.76  72.39            0.00           0.00
onset (y_t - y_t-3 > 30)               ridge  38518  93.19  61.44           15.10          15.12
onset (y_t - y_t-3 > 30) same_hour_yesterday  38518 109.76  72.39            0.00           0.00
onset (y_t - y_t-3 > 30)               stack  38518 102.67  65.58            6.46           9.40
        winter (Nov-Feb)           best_tree  69736 111.73  73.21           -2.88          -1.12
        winter (Nov-Feb)         climatology  69736 197.20 153.78          -81.57        -112.41
        winter (Nov-Feb)    equal_neural_ens  69736 105.00  68.25            3.32           5.73
        winter (Nov-Feb)         persistence  69736 108.61  72.40            0.00           0.00
        winter (Nov-Feb)               ridge  69736  99.65  67.46            8.25           6.82
        winter (Nov-Feb) same_hour_yesterday  69736 108.61  72.40            0.00           0.00
        winter (Nov-Feb)               stack  69736 107.15  69.73            1.34           3.69
  other months (Mar-Oct)           best_tree 131913  45.52  26.78            8.73           8.69
  other months (Mar-Oct)         climatology 131913  66.07  40.13          -32.48         -36.84
  other months (Mar-Oct)    equal_neural_ens 131913  44.80  26.67           10.18           9.07
  other months (Mar-Oct)         persistence 131913  49.87  29.32            0.00           0.00
  other months (Mar-Oct)               ridge 131913  43.91  26.67           11.96           9.05
  other months (Mar-Oct) same_hour_yesterday 131913  49.87  29.32            0.00           0.00
  other months (Mar-Oct)               stack 131913  45.61  26.87            8.54           8.38
                     all           best_tree 201649  75.32  42.83            0.29           3.14
                     all         climatology 201649 127.69  79.43          -69.03         -79.63
                     all    equal_neural_ens 201649  71.59  41.05            5.22           7.18
                     all         persistence 201649  75.54  44.22            0.00           0.00
                     all               ridge 201649  68.52  40.78            9.29           7.79
                     all same_hour_yesterday 201649  75.54  44.22            0.00           0.00
                     all               stack 201649  73.02  41.69            3.34           5.72
```

### `P1_LOCO_Mumbai_h1_direct_none`
```text
                  regime            model      N   RMSE    MAE  skill_RMSE_pct  skill_MAE_pct
               y_t >= 60        best_tree  34862  22.32  11.80            4.99           4.69
               y_t >= 60      climatology  34862  48.57  31.28         -106.77        -152.54
               y_t >= 60 equal_neural_ens  34862  21.96  11.84            6.50           4.37
               y_t >= 60      persistence  34862  23.49  12.38            0.00           0.00
               y_t >= 60            ridge  34862  22.92  11.80            2.42           4.72
               y_t >= 60            stack  34862  21.98  11.83            6.42           4.45
              y_t >= 120        best_tree   5366  44.18  25.46            6.26           5.94
              y_t >= 120      climatology   5366 105.24  88.48         -123.30        -226.96
              y_t >= 120 equal_neural_ens   5366  43.13  25.48            8.48           5.84
              y_t >= 120      persistence   5366  47.13  27.06            0.00           0.00
              y_t >= 120            ridge   5366  46.02  25.43            2.35           6.03
              y_t >= 120            stack   5366  43.16  25.42            8.42           6.07
              y_t >= 250        best_tree    336 117.85  76.47           10.98          10.73
              y_t >= 250      climatology    336 226.32 200.95          -70.96        -134.60
              y_t >= 250 equal_neural_ens    336 112.09  75.53           15.33          11.83
              y_t >= 250      persistence    336 132.38  85.66            0.00           0.00
              y_t >= 250            ridge    336 130.81  81.52            1.19           4.84
              y_t >= 250            stack    336 112.30  75.17           15.17          12.24
onset (y_t - y_t-3 > 30)        best_tree   5227  41.27  23.44            7.48           5.82
onset (y_t - y_t-3 > 30)      climatology   5227  79.74  55.68          -78.77        -123.75
onset (y_t - y_t-3 > 30) equal_neural_ens   5227  39.91  23.44           10.52           5.81
onset (y_t - y_t-3 > 30)      persistence   5227  44.60  24.88            0.00           0.00
onset (y_t - y_t-3 > 30)            ridge   5227  43.62  23.75            2.21           4.57
onset (y_t - y_t-3 > 30)            stack   5227  39.96  23.40           10.41           5.97
        winter (Nov-Feb)        best_tree  43612  19.33   9.72            3.25           2.82
        winter (Nov-Feb)      climatology  43612  44.24  29.69         -121.39        -196.71
        winter (Nov-Feb) equal_neural_ens  43612  18.98   9.71            5.00           2.96
        winter (Nov-Feb)      persistence  43612  19.98  10.01            0.00           0.00
        winter (Nov-Feb)            ridge  43612  19.71   9.77            1.37           2.33
        winter (Nov-Feb)            stack  43612  19.01   9.70            4.86           3.08
  other months (Mar-Oct)        best_tree  80813   8.91   4.60            7.89           3.39
  other months (Mar-Oct)      climatology  80813  45.08  41.03         -366.08        -761.68
  other months (Mar-Oct) equal_neural_ens  80813   8.91   4.59            7.92           3.64
  other months (Mar-Oct)      persistence  80813   9.67   4.76            0.00           0.00
  other months (Mar-Oct)            ridge  80813   9.39   4.81            2.89          -1.06
  other months (Mar-Oct)            stack  80813   8.90   4.59            7.97           3.52
                     all        best_tree 124425  13.51   6.40            4.63           3.09
                     all      climatology 124425  44.79  37.05         -216.11        -461.45
                     all equal_neural_ens 124425  13.34   6.38            5.88           3.28
                     all      persistence 124425  14.17   6.60            0.00           0.00
                     all            ridge 124425  13.91   6.55            1.83           0.74
                     all            stack 124425  13.35   6.38            5.79           3.29
```

### `P1_LOCO_Mumbai_h1_direct_openmeteo`
```text
                  regime            model      N   RMSE    MAE  skill_RMSE_pct  skill_MAE_pct
               y_t >= 60        best_tree  34862  21.88  11.83            6.86           4.52
               y_t >= 60      climatology  34862  48.57  31.28         -106.77        -152.54
               y_t >= 60 equal_neural_ens  34862  21.79  11.86            7.23           4.25
               y_t >= 60      persistence  34862  23.49  12.38            0.00           0.00
               y_t >= 60            ridge  34862  22.78  11.84            3.00           4.42
               y_t >= 60            stack  34862  21.86  11.85            6.94           4.31
              y_t >= 120        best_tree   5366  42.85  25.09            9.07           7.28
              y_t >= 120      climatology   5366 105.24  88.48         -123.30        -226.96
              y_t >= 120 equal_neural_ens   5366  42.55  25.04            9.71           7.46
              y_t >= 120      persistence   5366  47.13  27.06            0.00           0.00
              y_t >= 120            ridge   5366  45.58  25.24            3.28           6.73
              y_t >= 120            stack   5366  42.72  24.92            9.36           7.91
              y_t >= 250        best_tree    336 112.15  73.51           15.28          14.18
              y_t >= 250      climatology    336 226.32 200.95          -70.96        -134.60
              y_t >= 250 equal_neural_ens    336 111.51  73.90           15.77          13.73
              y_t >= 250      persistence    336 132.38  85.66            0.00           0.00
              y_t >= 250            ridge    336 128.96  80.47            2.59           6.06
              y_t >= 250            stack    336 112.73  74.18           14.84          13.40
onset (y_t - y_t-3 > 30)        best_tree   5227  39.72  22.90           10.95           7.99
onset (y_t - y_t-3 > 30)      climatology   5227  79.74  55.68          -78.77        -123.75
onset (y_t - y_t-3 > 30) equal_neural_ens   5227  39.23  23.01           12.05           7.51
onset (y_t - y_t-3 > 30)      persistence   5227  44.60  24.88            0.00           0.00
onset (y_t - y_t-3 > 30)            ridge   5227  43.19  23.67            3.16           4.89
onset (y_t - y_t-3 > 30)            stack   5227  39.60  23.02           11.23           7.49
        winter (Nov-Feb)        best_tree  43612  18.87   9.74            5.59           2.65
        winter (Nov-Feb)      climatology  43612  44.24  29.69         -121.39        -196.71
        winter (Nov-Feb) equal_neural_ens  43612  18.81   9.73            5.86           2.75
        winter (Nov-Feb)      persistence  43612  19.98  10.01            0.00           0.00
        winter (Nov-Feb)            ridge  43612  19.61   9.87            1.85           1.40
        winter (Nov-Feb)            stack  43612  18.87   9.75            5.59           2.59
  other months (Mar-Oct)        best_tree  80813   8.92   4.62            7.81           3.02
  other months (Mar-Oct)      climatology  80813  45.08  41.03         -366.08        -761.68
  other months (Mar-Oct) equal_neural_ens  80813   8.93   4.65            7.65           2.43
  other months (Mar-Oct)      persistence  80813   9.67   4.76            0.00           0.00
  other months (Mar-Oct)            ridge  80813   9.50   5.08            1.78          -6.78
  other months (Mar-Oct)            stack  80813   8.95   4.66            7.42           2.10
                     all        best_tree 124425  13.28   6.41            6.25           2.83
                     all      climatology 124425  44.79  37.05         -216.11        -461.45
                     all equal_neural_ens 124425  13.26   6.43            6.40           2.60
                     all      persistence 124425  14.17   6.60            0.00           0.00
                     all            ridge 124425  13.91   6.76            1.83          -2.44
                     all            stack 124425  13.30   6.44            6.14           2.36
```

### `P1_LOCO_Kolkata_h1_direct_none`
```text
                  regime            model      N   RMSE    MAE  skill_RMSE_pct  skill_MAE_pct
               y_t >= 60        best_tree  54885  14.67   8.11           13.40          13.03
               y_t >= 60      climatology  54885  71.63  50.66         -322.91        -443.15
               y_t >= 60 equal_neural_ens  54885  14.31   7.84           15.50          15.94
               y_t >= 60      persistence  54885  16.94   9.33            0.00           0.00
               y_t >= 60            ridge  54885  14.89   8.13           12.07          12.82
               y_t >= 60            stack  54885  14.29   7.85           15.66          15.82
              y_t >= 120        best_tree  17829  21.53  11.64           14.73          18.29
              y_t >= 120      climatology  17829 117.12 102.52         -363.75        -619.73
              y_t >= 120 equal_neural_ens  17829  21.09  11.36           16.47          20.26
              y_t >= 120      persistence  17829  25.25  14.24            0.00           0.00
              y_t >= 120            ridge  17829  22.00  11.97           12.89          16.00
              y_t >= 120            stack  17829  21.04  11.31           16.70          20.58
              y_t >= 250        best_tree   1378  52.63  30.33           11.73          17.73
              y_t >= 250      climatology   1378 253.15 240.57         -324.59        -552.55
              y_t >= 250 equal_neural_ens   1378  51.25  29.85           14.05          19.04
              y_t >= 250      persistence   1378  59.62  36.87            0.00           0.00
              y_t >= 250            ridge   1378  52.52  31.11           11.92          15.61
              y_t >= 250            stack   1378  51.10  29.63           14.30          19.63
onset (y_t - y_t-3 > 30)        best_tree   7551  28.58  14.98           12.22          18.18
onset (y_t - y_t-3 > 30)      climatology   7551 125.92  99.73         -286.82        -444.70
onset (y_t - y_t-3 > 30) equal_neural_ens   7551  27.89  14.59           14.33          20.31
onset (y_t - y_t-3 > 30)      persistence   7551  32.55  18.31            0.00           0.00
onset (y_t - y_t-3 > 30)            ridge   7551  29.33  15.94            9.91          12.93
onset (y_t - y_t-3 > 30)            stack   7551  27.74  14.44           14.79          21.15
        winter (Nov-Feb)        best_tree  51582  13.75   7.80           14.48          12.86
        winter (Nov-Feb)      climatology  51582  72.19  51.34         -349.05        -473.61
        winter (Nov-Feb) equal_neural_ens  51582  13.50   7.46           16.05          16.69
        winter (Nov-Feb)      persistence  51582  16.08   8.95            0.00           0.00
        winter (Nov-Feb)            ridge  51582  14.28   7.97           11.17          11.00
        winter (Nov-Feb)            stack  51582  13.47   7.48           16.21          16.45
  other months (Mar-Oct)        best_tree 103951   6.90   3.57            4.46          -2.25
  other months (Mar-Oct)      climatology 103951  40.17  35.31         -456.02        -910.85
  other months (Mar-Oct) equal_neural_ens 103951   6.78   3.59            6.19          -2.82
  other months (Mar-Oct)      persistence 103951   7.22   3.49            0.00           0.00
  other months (Mar-Oct)            ridge 103951   6.87   3.65            4.89          -4.54
  other months (Mar-Oct)            stack 103951   6.78   3.62            6.12          -3.63
                     all        best_tree 155533   9.72   4.97           11.46           6.21
                     all      climatology 155533  52.98  40.63         -382.44        -666.11
                     all equal_neural_ens 155533   9.55   4.87           13.08           8.10
                     all      persistence 155533  10.98   5.30            0.00           0.00
                     all            ridge 155533   9.96   5.08            9.31           4.16
                     all            stack 155533   9.54   4.90           13.17           7.61
```

### `P1_LOCO_Kolkata_h1_direct_openmeteo`
```text
                  regime            model      N   RMSE    MAE  skill_RMSE_pct  skill_MAE_pct
               y_t >= 60        best_tree  54885  15.62   9.02            7.76           3.24
               y_t >= 60      climatology  54885  71.63  50.66         -322.91        -443.15
               y_t >= 60 equal_neural_ens  54885  14.62   8.46           13.67           9.25
               y_t >= 60      persistence  54885  16.94   9.33            0.00           0.00
               y_t >= 60            ridge  54885  14.74   8.12           12.97          12.97
               y_t >= 60            stack  54885  14.54   8.36           14.18          10.33
              y_t >= 120        best_tree  17829  23.01  13.26            8.89           6.91
              y_t >= 120      climatology  17829 117.12 102.52         -363.75        -619.73
              y_t >= 120 equal_neural_ens  17829  21.48  12.24           14.95          14.08
              y_t >= 120      persistence  17829  25.25  14.24            0.00           0.00
              y_t >= 120            ridge  17829  21.76  11.75           13.82          17.53
              y_t >= 120            stack  17829  21.32  11.98           15.58          15.88
              y_t >= 250        best_tree   1378  54.64  33.83            8.35           8.24
              y_t >= 250      climatology   1378 253.15 240.57         -324.59        -552.55
              y_t >= 250 equal_neural_ens   1378  51.09  30.67           14.31          16.79
              y_t >= 250      persistence   1378  59.62  36.87            0.00           0.00
              y_t >= 250            ridge   1378  52.43  31.13           12.06          15.56
              y_t >= 250            stack   1378  51.16  30.18           14.19          18.15
onset (y_t - y_t-3 > 30)        best_tree   7551  29.34  15.70            9.88          14.24
onset (y_t - y_t-3 > 30)      climatology   7551 125.92  99.73         -286.82        -444.70
onset (y_t - y_t-3 > 30) equal_neural_ens   7551  27.51  14.36           15.49          21.58
onset (y_t - y_t-3 > 30)      persistence   7551  32.55  18.31            0.00           0.00
onset (y_t - y_t-3 > 30)            ridge   7551  28.97  15.48           10.99          15.43
onset (y_t - y_t-3 > 30)            stack   7551  27.49  14.23           15.54          22.26
        winter (Nov-Feb)        best_tree  51582  14.92   8.90            7.22           0.51
        winter (Nov-Feb)      climatology  51582  72.19  51.34         -349.05        -473.61
        winter (Nov-Feb) equal_neural_ens  51582  13.89   8.25           13.61           7.82
        winter (Nov-Feb)      persistence  51582  16.08   8.95            0.00           0.00
        winter (Nov-Feb)            ridge  51582  14.14   8.03           12.03          10.25
        winter (Nov-Feb)            stack  51582  13.79   8.20           14.22           8.39
  other months (Mar-Oct)        best_tree 103951   6.90   3.59            4.44          -2.75
  other months (Mar-Oct)      climatology 103951  40.17  35.31         -456.02        -910.85
  other months (Mar-Oct) equal_neural_ens 103951   6.83   3.66            5.47          -4.78
  other months (Mar-Oct)      persistence 103951   7.22   3.49            0.00           0.00
  other months (Mar-Oct)            ridge 103951   7.07   3.90            2.21         -11.68
  other months (Mar-Oct)            stack 103951   6.84   3.65            5.32          -4.49
                     all        best_tree 155533  10.28   5.35            6.41          -0.93
                     all      climatology 155533  52.98  40.63         -382.44        -666.11
                     all equal_neural_ens 155533   9.75   5.18           11.18           2.27
                     all      persistence 155533  10.98   5.30            0.00           0.00
                     all            ridge 155533   9.98   5.27            9.08           0.60
                     all            stack 155533   9.71   5.16           11.55           2.72
```

### `P1_LOCO_Chennai_h1_direct_none`
```text
                  regime            model      N   RMSE    MAE  skill_RMSE_pct  skill_MAE_pct
               y_t >= 60        best_tree  15680  44.95  20.88            7.38           9.82
               y_t >= 60      climatology  15680  73.41  36.30          -51.23         -56.80
               y_t >= 60 equal_neural_ens  15680  44.39  20.78            8.55          10.27
               y_t >= 60      persistence  15680  48.54  23.15            0.00           0.00
               y_t >= 60            ridge  15680  46.46  21.56            4.28           6.88
               y_t >= 60            stack  15680  44.04  20.62            9.26          10.95
              y_t >= 120        best_tree   2510  85.33  50.98           10.75          13.25
              y_t >= 120      climatology   2510 161.55 109.24          -68.98         -85.91
              y_t >= 120 equal_neural_ens   2510  83.49  49.80           12.67          15.25
              y_t >= 120      persistence   2510  95.60  58.76            0.00           0.00
              y_t >= 120            ridge   2510  89.61  53.01            6.26           9.80
              y_t >= 120            stack   2510  82.46  49.29           13.75          16.12
              y_t >= 250        best_tree    421 152.04 111.76           15.50          18.28
              y_t >= 250      climatology    421 316.58 253.06          -75.95         -85.05
              y_t >= 250 equal_neural_ens    421 148.47 106.92           17.48          21.82
              y_t >= 250      persistence    421 179.92 136.75            0.00           0.00
              y_t >= 250            ridge    421 167.36 123.90            6.98           9.40
              y_t >= 250            stack    421 145.64 105.47           19.06          22.88
onset (y_t - y_t-3 > 30)        best_tree   5796  57.19  31.19            9.19          10.25
onset (y_t - y_t-3 > 30)      climatology   5796 101.58  53.55          -61.28         -54.08
onset (y_t - y_t-3 > 30) equal_neural_ens   5796  56.78  31.19            9.85          10.24
onset (y_t - y_t-3 > 30)      persistence   5796  62.99  34.75            0.00           0.00
onset (y_t - y_t-3 > 30)            ridge   5796  60.38  32.84            4.13           5.52
onset (y_t - y_t-3 > 30)            stack   5796  56.14  30.96           10.86          10.91
        winter (Nov-Feb)        best_tree  39944  22.87   8.33            6.91           5.05
        winter (Nov-Feb)      climatology  39944  49.61  36.33         -101.97        -314.26
        winter (Nov-Feb) equal_neural_ens  39944  22.77   8.32            7.32           5.16
        winter (Nov-Feb)      persistence  39944  24.56   8.77            0.00           0.00
        winter (Nov-Feb)            ridge  39944  23.69   8.82            3.55          -0.62
        winter (Nov-Feb)            stack  39944  22.57   8.28            8.13           5.59
  other months (Mar-Oct)        best_tree  78141  17.49   6.90            5.47           5.00
  other months (Mar-Oct)      climatology  78141  51.63  44.65         -178.98        -514.37
  other months (Mar-Oct) equal_neural_ens  78141  17.23   6.86            6.90           5.69
  other months (Mar-Oct)      persistence  78141  18.51   7.27            0.00           0.00
  other months (Mar-Oct)            ridge  78141  18.03   7.17            2.59           1.38
  other months (Mar-Oct)            stack  78141  17.17   6.85            7.23           5.73
                     all        best_tree 118085  19.48   7.39            6.15           5.02
                     all      climatology 118085  50.95  41.84         -145.52        -438.04
                     all equal_neural_ens 118085  19.28   7.35            7.10           5.49
                     all      persistence 118085  20.75   7.78            0.00           0.00
                     all            ridge 118085  20.12   7.73            3.05           0.61
                     all            stack 118085  19.17   7.33            7.65           5.68
```

### `P1_LOCO_Chennai_h1_direct_openmeteo`
```text
                  regime            model      N   RMSE    MAE  skill_RMSE_pct  skill_MAE_pct
               y_t >= 60        best_tree  15680  44.82  20.41            7.67          11.86
               y_t >= 60      climatology  15680  73.41  36.30          -51.23         -56.80
               y_t >= 60 equal_neural_ens  15680  45.63  20.87            6.00           9.84
               y_t >= 60      persistence  15680  48.54  23.15            0.00           0.00
               y_t >= 60            ridge  15680  46.12  21.35            4.99           7.79
               y_t >= 60            stack  15680  45.30  20.80            6.67          10.18
              y_t >= 120        best_tree   2510  85.08  48.83           11.00          16.90
              y_t >= 120      climatology   2510 161.55 109.24          -68.98         -85.91
              y_t >= 120 equal_neural_ens   2510  87.50  49.83            8.47          15.20
              y_t >= 120      persistence   2510  95.60  58.76            0.00           0.00
              y_t >= 120            ridge   2510  88.74  52.38            7.17          10.86
              y_t >= 120            stack   2510  86.59  49.65            9.43          15.51
              y_t >= 250        best_tree    421 153.73 107.06           14.56          21.72
              y_t >= 250      climatology    421 316.58 253.06          -75.95         -85.05
              y_t >= 250 equal_neural_ens    421 160.96 109.90           10.54          19.64
              y_t >= 250      persistence    421 179.92 136.75            0.00           0.00
              y_t >= 250            ridge    421 165.14 122.28            8.22          10.59
              y_t >= 250            stack    421 158.32 109.13           12.01          20.20
onset (y_t - y_t-3 > 30)        best_tree   5796  57.53  30.21            8.66          13.06
onset (y_t - y_t-3 > 30)      climatology   5796 101.58  53.55          -61.28         -54.08
onset (y_t - y_t-3 > 30) equal_neural_ens   5796  59.17  30.79            6.06          11.40
onset (y_t - y_t-3 > 30)      persistence   5796  62.99  34.75            0.00           0.00
onset (y_t - y_t-3 > 30)            ridge   5796  59.71  32.37            5.20           6.85
onset (y_t - y_t-3 > 30)            stack   5796  58.51  30.70            7.10          11.66
        winter (Nov-Feb)        best_tree  39944  22.88   8.23            6.87           6.10
        winter (Nov-Feb)      climatology  39944  49.61  36.33         -101.97        -314.26
        winter (Nov-Feb) equal_neural_ens  39944  23.62   8.41            3.84           4.05
        winter (Nov-Feb)      persistence  39944  24.56   8.77            0.00           0.00
        winter (Nov-Feb)            ridge  39944  23.52   8.64            4.24           1.52
        winter (Nov-Feb)            stack  39944  23.48   8.37            4.43           4.54
  other months (Mar-Oct)        best_tree  78141  17.47   6.79            5.60           6.53
  other months (Mar-Oct)      climatology  78141  51.63  44.65         -178.98        -514.37
  other months (Mar-Oct) equal_neural_ens  78141  17.39   6.87            6.04           5.41
  other months (Mar-Oct)      persistence  78141  18.51   7.27            0.00           0.00
  other months (Mar-Oct)            ridge  78141  17.96   7.19            2.96           1.10
  other months (Mar-Oct)            stack  78141  17.29   6.84            6.56           5.94
                     all        best_tree 118085  19.47   7.28            6.20           6.37
                     all      climatology 118085  50.95  41.84         -145.52        -438.04
                     all equal_neural_ens 118085  19.72   7.40            4.99           4.90
                     all      persistence 118085  20.75   7.78            0.00           0.00
                     all            ridge 118085  20.01   7.68            3.56           1.26
                     all            stack 118085  19.60   7.36            5.55           5.41
```

### `P1_LOCO_Bengaluru_h1_direct_none`
```text
                  regime            model      N   RMSE    MAE  skill_RMSE_pct  skill_MAE_pct
               y_t >= 60        best_tree  12326  33.08  14.11           12.76          15.97
               y_t >= 60      climatology  12326  52.09  24.27          -37.37         -44.55
               y_t >= 60 equal_neural_ens  12326  32.74  14.10           13.64          16.00
               y_t >= 60      persistence  12326  37.92  16.79            0.00           0.00
               y_t >= 60            ridge  12326  34.51  14.79            8.99          11.92
               y_t >= 60            stack  12326  32.64  14.02           13.91          16.48
              y_t >= 120        best_tree   1088  96.86  54.57           14.30          22.28
              y_t >= 120      climatology   1088 157.90 100.78          -39.71         -43.54
              y_t >= 120 equal_neural_ens   1088  95.84  53.39           15.20          23.96
              y_t >= 120      persistence   1088 113.02  70.21            0.00           0.00
              y_t >= 120            ridge   1088 102.12  58.95            9.64          16.04
              y_t >= 120            stack   1088  95.44  52.79           15.55          24.82
              y_t >= 250        best_tree    188 181.63 129.91           17.66          25.49
              y_t >= 250      climatology    188 301.96 230.19          -36.89         -32.02
              y_t >= 250 equal_neural_ens    188 178.11 122.11           19.26          29.97
              y_t >= 250      persistence    188 220.59 174.36            0.00           0.00
              y_t >= 250            ridge    188 198.81 149.57            9.88          14.22
              y_t >= 250            stack    188 176.43 119.90           20.02          31.24
onset (y_t - y_t-3 > 30)        best_tree   3951  53.75  24.55           12.68          17.09
onset (y_t - y_t-3 > 30)      climatology   3951  85.43  42.51          -38.80         -43.54
onset (y_t - y_t-3 > 30) equal_neural_ens   3951  53.23  24.54           13.51          17.15
onset (y_t - y_t-3 > 30)      persistence   3951  61.55  29.62            0.00           0.00
onset (y_t - y_t-3 > 30)            ridge   3951  56.52  26.56            8.17          10.32
onset (y_t - y_t-3 > 30)            stack   3951  53.08  24.39           13.75          17.64
        winter (Nov-Feb)        best_tree  57935  15.10   5.65            8.87           8.34
        winter (Nov-Feb)      climatology  57935  42.47  35.72         -156.37        -479.23
        winter (Nov-Feb) equal_neural_ens  57935  14.88   5.63           10.17           8.77
        winter (Nov-Feb)      persistence  57935  16.57   6.17            0.00           0.00
        winter (Nov-Feb)            ridge  57935  15.71   6.05            5.15           1.85
        winter (Nov-Feb)            stack  57935  14.87   5.63           10.23           8.74
  other months (Mar-Oct)        best_tree 110803  10.79   4.14            8.55           6.29
  other months (Mar-Oct)      climatology 110803  48.89  44.63         -314.47        -910.22
  other months (Mar-Oct) equal_neural_ens 110803  10.81   4.13            8.34           6.44
  other months (Mar-Oct)      persistence 110803  11.80   4.42            0.00           0.00
  other months (Mar-Oct)            ridge 110803  11.17   4.43            5.33          -0.34
  other months (Mar-Oct)            stack 110803  10.78   4.12            8.63           6.78
                     all        best_tree 168738  12.44   4.66            8.71           7.16
                     all      climatology 168738  46.79  41.57         -243.42        -728.40
                     all equal_neural_ens 168738  12.36   4.65            9.27           7.42
                     all      persistence 168738  13.62   5.02            0.00           0.00
                     all            ridge 168738  12.91   4.99            5.24           0.58
                     all            stack 168738  12.34   4.64            9.44           7.61
```

### `P1_LOCO_Bengaluru_h1_direct_openmeteo`
```text
                  regime            model      N   RMSE    MAE  skill_RMSE_pct  skill_MAE_pct
               y_t >= 60        best_tree  12326  33.33  14.35           12.09          14.54
               y_t >= 60      climatology  12326  52.09  24.27          -37.37         -44.55
               y_t >= 60 equal_neural_ens  12326  34.16  17.36            9.92          -3.42
               y_t >= 60      persistence  12326  37.92  16.79            0.00           0.00
               y_t >= 60            ridge  12326  34.58  15.48            8.80           7.80
               y_t >= 60            stack  12326  33.36  16.39           12.02           2.39
              y_t >= 120        best_tree   1088  97.81  56.28           13.46          19.84
              y_t >= 120      climatology   1088 157.90 100.78          -39.71         -43.54
              y_t >= 120 equal_neural_ens   1088  96.58  55.93           14.55          20.34
              y_t >= 120      persistence   1088 113.02  70.21            0.00           0.00
              y_t >= 120            ridge   1088 101.54  59.33           10.15          15.50
              y_t >= 120            stack   1088  95.05  55.93           15.89          20.34
              y_t >= 250        best_tree    188 182.19 134.75           17.41          22.72
              y_t >= 250      climatology    188 301.96 230.19          -36.89         -32.02
              y_t >= 250 equal_neural_ens    188 177.20 127.48           19.67          26.89
              y_t >= 250      persistence    188 220.59 174.36            0.00           0.00
              y_t >= 250            ridge    188 196.38 147.97           10.98          15.13
              y_t >= 250            stack    188 173.80 126.08           21.21          27.69
onset (y_t - y_t-3 > 30)        best_tree   3951  53.91  24.96           12.41          15.72
onset (y_t - y_t-3 > 30)      climatology   3951  85.43  42.51          -38.80         -43.54
onset (y_t - y_t-3 > 30) equal_neural_ens   3951  54.14  27.49           12.03           7.17
onset (y_t - y_t-3 > 30)      persistence   3951  61.55  29.62            0.00           0.00
onset (y_t - y_t-3 > 30)            ridge   3951  56.36  27.33            8.42           7.71
onset (y_t - y_t-3 > 30)            stack   3951  53.15  26.62           13.64          10.12
        winter (Nov-Feb)        best_tree  57935  15.05   5.74            9.19           6.99
        winter (Nov-Feb)      climatology  57935  42.47  35.72         -156.37        -479.23
        winter (Nov-Feb) equal_neural_ens  57935  18.41  12.25          -11.15         -98.73
        winter (Nov-Feb)      persistence  57935  16.57   6.17            0.00           0.00
        winter (Nov-Feb)            ridge  57935  15.87   6.66            4.22          -7.96
        winter (Nov-Feb)            stack  57935  16.57   9.51            0.01         -54.26
  other months (Mar-Oct)        best_tree 110803  10.96   4.28            7.06           3.03
  other months (Mar-Oct)      climatology 110803  48.89  44.63         -314.47        -910.22
  other months (Mar-Oct) equal_neural_ens 110803  15.89  12.24          -34.72        -177.15
  other months (Mar-Oct)      persistence 110803  11.80   4.42            0.00           0.00
  other months (Mar-Oct)            ridge 110803  11.69   5.70            0.92         -29.07
  other months (Mar-Oct)            stack 110803  13.12   8.54          -11.18         -93.29
                     all        best_tree 168738  12.52   4.78            8.14           4.70
                     all      climatology 168738  46.79  41.57         -243.42        -728.40
                     all equal_neural_ens 168738  16.80  12.25          -23.31        -144.06
                     all      persistence 168738  13.62   5.02            0.00           0.00
                     all            ridge 168738  13.27   6.03            2.58         -20.17
                     all            stack 168738  14.39   8.87           -5.65         -76.83
```

