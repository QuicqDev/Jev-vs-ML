window.BENCHMARK = {
  "raw": {
    "models": [
      "Logistic regression",
      "SVM",
      "Decision tree",
      "Random forest",
      "Extra trees",
      "k-NN",
      "Naive Bayes",
      "Hist gradient boost",
      "XGBoost",
      "CatBoost",
      "Voting ensemble",
      "Majority baseline",
      "Jev zero-shot",
      "Jev few-shot"
    ],
    "rows": [
      {
        "dataset": "AG News",
        "kind": "text",
        "testRows": 1000,
        "scores": {
          "Logistic regression": {
            "mean": 87.4,
            "sd": 0.8,
            "n": 3,
            "display": "87.4 ± 0.8 (n=3)"
          },
          "SVM": {
            "mean": 88.4,
            "sd": 0.3,
            "n": 3,
            "display": "88.4 ± 0.3 (n=3)"
          },
          "Decision tree": {
            "mean": 67.5,
            "sd": 0.9,
            "n": 3,
            "display": "67.5 ± 0.9 (n=3)"
          },
          "Random forest": {
            "mean": 71.6,
            "sd": 1.1,
            "n": 3,
            "display": "71.6 ± 1.1 (n=3)"
          },
          "Extra trees": {
            "mean": 75.1,
            "sd": 0.7,
            "n": 3,
            "display": "75.1 ± 0.7 (n=3)"
          },
          "k-NN": {
            "mean": 78.4,
            "sd": 0.3,
            "n": 3,
            "display": "78.4 ± 0.3 (n=3)"
          },
          "Naive Bayes": {
            "mean": 87.4,
            "sd": 0.2,
            "n": 3,
            "display": "87.4 ± 0.2 (n=3)"
          },
          "Hist gradient boost": {
            "mean": 80.3,
            "sd": 0.8,
            "n": 3,
            "display": "80.3 ± 0.8 (n=3)"
          },
          "XGBoost": {
            "mean": 82.6,
            "sd": 0.2,
            "n": 3,
            "display": "82.6 ± 0.2 (n=3)"
          },
          "CatBoost": {
            "mean": 82.0,
            "sd": 1.0,
            "n": 3,
            "display": "82.0 ± 1.0 (n=3)"
          },
          "Voting ensemble": {
            "mean": 87.0,
            "sd": 1.0,
            "n": 3,
            "display": "87.0 ± 1.0 (n=3)"
          },
          "Majority baseline": {
            "mean": 25.0,
            "sd": 0.0,
            "n": 3,
            "display": "25.0 ± 0.0 (n=3)"
          },
          "Jev zero-shot": {
            "mean": 87.5,
            "sd": 0.0,
            "n": 3,
            "display": "87.5 ± 0.0 (n=3)"
          },
          "Jev few-shot": {
            "mean": 86.3,
            "sd": 0.6,
            "n": 3,
            "display": "86.3 ± 0.6 (n=3)"
          }
        },
        "best": 88.4,
        "bestModels": [
          "SVM"
        ]
      },
      {
        "dataset": "Banking77",
        "kind": "text",
        "testRows": 1500,
        "scores": {
          "Logistic regression": {
            "mean": 89.4,
            "sd": 0.2,
            "n": 3,
            "display": "89.4 ± 0.2 (n=3)"
          },
          "SVM": {
            "mean": 89.7,
            "sd": 0.6,
            "n": 3,
            "display": "89.7 ± 0.6 (n=3)"
          },
          "Decision tree": {
            "mean": 62.2,
            "sd": 0.9,
            "n": 3,
            "display": "62.2 ± 0.9 (n=3)"
          },
          "Random forest": {
            "mean": 68.2,
            "sd": 1.4,
            "n": 3,
            "display": "68.2 ± 1.4 (n=3)"
          },
          "Extra trees": {
            "mean": 69.8,
            "sd": 0.4,
            "n": 3,
            "display": "69.8 ± 0.4 (n=3)"
          },
          "k-NN": {
            "mean": 58.4,
            "sd": 1.9,
            "n": 3,
            "display": "58.4 ± 1.9 (n=3)"
          },
          "Naive Bayes": {
            "mean": 86.0,
            "sd": 0.5,
            "n": 3,
            "display": "86.0 ± 0.5 (n=3)"
          },
          "Hist gradient boost": {
            "mean": 62.2,
            "sd": 1.2,
            "n": 3,
            "display": "62.2 ± 1.2 (n=3)"
          },
          "XGBoost": {
            "mean": 71.9,
            "sd": 0.4,
            "n": 3,
            "display": "71.9 ± 0.4 (n=3)"
          },
          "CatBoost": {
            "mean": 68.1,
            "sd": 0.7,
            "n": 3,
            "display": "68.1 ± 0.7 (n=3)"
          },
          "Voting ensemble": {
            "mean": 86.2,
            "sd": 0.4,
            "n": 3,
            "display": "86.2 ± 0.4 (n=3)"
          },
          "Majority baseline": {
            "mean": 1.3,
            "sd": 0.0,
            "n": 3,
            "display": "1.3 ± 0.0 (n=3)"
          },
          "Jev zero-shot": {
            "mean": 78.9,
            "sd": 0.0,
            "n": 3,
            "display": "78.9 ± 0.0 (n=3)"
          },
          "Jev few-shot": {
            "mean": 81.9,
            "sd": 1.7,
            "n": 3,
            "display": "81.9 ± 1.7 (n=3)"
          }
        },
        "best": 89.7,
        "bestModels": [
          "SVM"
        ]
      },
      {
        "dataset": "SMS Spam",
        "kind": "text",
        "testRows": 1000,
        "scores": {
          "Logistic regression": {
            "mean": 86.4,
            "sd": 8.1,
            "n": 3,
            "display": "86.4 ± 8.1 (n=3)"
          },
          "SVM": {
            "mean": 93.7,
            "sd": 0.0,
            "n": 3,
            "display": "93.7 ± 0.0 (n=3)"
          },
          "Decision tree": {
            "mean": 85.7,
            "sd": 2.8,
            "n": 3,
            "display": "85.7 ± 2.8 (n=3)"
          },
          "Random forest": {
            "mean": 89.0,
            "sd": 0.3,
            "n": 3,
            "display": "89.0 ± 0.3 (n=3)"
          },
          "Extra trees": {
            "mean": 88.7,
            "sd": 0.8,
            "n": 3,
            "display": "88.7 ± 0.8 (n=3)"
          },
          "k-NN": {
            "mean": 87.8,
            "sd": 2.8,
            "n": 3,
            "display": "87.8 ± 2.8 (n=3)"
          },
          "Naive Bayes": {
            "mean": 95.0,
            "sd": 1.9,
            "n": 3,
            "display": "95.0 ± 1.9 (n=3)"
          },
          "Hist gradient boost": {
            "mean": 92.1,
            "sd": 0.1,
            "n": 3,
            "display": "92.1 ± 0.1 (n=3)"
          },
          "XGBoost": {
            "mean": 88.3,
            "sd": 2.8,
            "n": 3,
            "display": "88.3 ± 2.8 (n=3)"
          },
          "CatBoost": {
            "mean": 90.7,
            "sd": 1.6,
            "n": 3,
            "display": "90.7 ± 1.6 (n=3)"
          },
          "Voting ensemble": {
            "mean": 89.0,
            "sd": 0.8,
            "n": 3,
            "display": "89.0 ± 0.8 (n=3)"
          },
          "Majority baseline": {
            "mean": 50.0,
            "sd": 0.0,
            "n": 3,
            "display": "50.0 ± 0.0 (n=3)"
          },
          "Jev zero-shot": {
            "mean": 96.1,
            "sd": 0.0,
            "n": 3,
            "display": "96.1 ± 0.0 (n=3)"
          },
          "Jev few-shot": {
            "mean": 95.6,
            "sd": 0.9,
            "n": 3,
            "display": "95.6 ± 0.9 (n=3)"
          }
        },
        "best": 95.0,
        "bestModels": [
          "Naive Bayes"
        ]
      },
      {
        "dataset": "IMDb",
        "kind": "text",
        "testRows": 1000,
        "scores": {
          "Logistic regression": {
            "mean": 88.4,
            "sd": 0.2,
            "n": 3,
            "display": "88.4 ± 0.2 (n=3)"
          },
          "SVM": {
            "mean": 87.8,
            "sd": 0.9,
            "n": 3,
            "display": "87.8 ± 0.9 (n=3)"
          },
          "Decision tree": {
            "mean": 70.7,
            "sd": 1.1,
            "n": 3,
            "display": "70.7 ± 1.1 (n=3)"
          },
          "Random forest": {
            "mean": 81.2,
            "sd": 0.8,
            "n": 3,
            "display": "81.2 ± 0.8 (n=3)"
          },
          "Extra trees": {
            "mean": 83.3,
            "sd": 1.1,
            "n": 3,
            "display": "83.3 ± 1.1 (n=3)"
          },
          "k-NN": {
            "mean": 79.9,
            "sd": 1.5,
            "n": 3,
            "display": "79.9 ± 1.5 (n=3)"
          },
          "Naive Bayes": {
            "mean": 86.2,
            "sd": 0.4,
            "n": 3,
            "display": "86.2 ± 0.4 (n=3)"
          },
          "Hist gradient boost": {
            "mean": 83.2,
            "sd": 0.3,
            "n": 3,
            "display": "83.2 ± 0.3 (n=3)"
          },
          "XGBoost": {
            "mean": 83.6,
            "sd": 0.2,
            "n": 3,
            "display": "83.6 ± 0.2 (n=3)"
          },
          "CatBoost": {
            "mean": 84.0,
            "sd": 0.2,
            "n": 3,
            "display": "84.0 ± 0.2 (n=3)"
          },
          "Voting ensemble": {
            "mean": 87.3,
            "sd": 0.2,
            "n": 3,
            "display": "87.3 ± 0.2 (n=3)"
          },
          "Majority baseline": {
            "mean": 50.0,
            "sd": 0.0,
            "n": 3,
            "display": "50.0 ± 0.0 (n=3)"
          },
          "Jev zero-shot": {
            "mean": 96.3,
            "sd": 0.0,
            "n": 3,
            "display": "96.3 ± 0.0 (n=3)"
          },
          "Jev few-shot": {
            "mean": 95.9,
            "sd": 0.5,
            "n": 3,
            "display": "95.9 ± 0.5 (n=3)"
          }
        },
        "best": 88.4,
        "bestModels": [
          "Logistic regression"
        ]
      },
      {
        "dataset": "Bank Marketing",
        "kind": "tabular",
        "testRows": 1000,
        "scores": {
          "Logistic regression": {
            "mean": 58.0,
            "sd": 0.6,
            "n": 3,
            "display": "58.0 ± 0.6 (n=3)"
          },
          "SVM": {
            "mean": 71.8,
            "sd": 0.7,
            "n": 3,
            "display": "71.8 ± 0.7 (n=3)"
          },
          "Decision tree": {
            "mean": 66.0,
            "sd": 4.8,
            "n": 3,
            "display": "66.0 ± 4.8 (n=3)"
          },
          "Random forest": {
            "mean": 59.3,
            "sd": 1.4,
            "n": 3,
            "display": "59.3 ± 1.4 (n=3)"
          },
          "Extra trees": {
            "mean": 60.2,
            "sd": 0.3,
            "n": 3,
            "display": "60.2 ± 0.3 (n=3)"
          },
          "k-NN": {
            "mean": 55.5,
            "sd": 0.5,
            "n": 3,
            "display": "55.5 ± 0.5 (n=3)"
          },
          "Naive Bayes": {
            "mean": 71.0,
            "sd": 0.4,
            "n": 3,
            "display": "71.0 ± 0.4 (n=3)"
          },
          "Hist gradient boost": {
            "mean": 63.9,
            "sd": 7.4,
            "n": 3,
            "display": "63.9 ± 7.4 (n=3)"
          },
          "XGBoost": {
            "mean": 64.1,
            "sd": 7.0,
            "n": 3,
            "display": "64.1 ± 7.0 (n=3)"
          },
          "CatBoost": {
            "mean": 62.3,
            "sd": 7.8,
            "n": 3,
            "display": "62.3 ± 7.8 (n=3)"
          },
          "Voting ensemble": {
            "mean": 61.4,
            "sd": 5.5,
            "n": 3,
            "display": "61.4 ± 5.5 (n=3)"
          },
          "Majority baseline": {
            "mean": 50.0,
            "sd": 0.0,
            "n": 3,
            "display": "50.0 ± 0.0 (n=3)"
          },
          "Jev zero-shot": {
            "mean": 53.4,
            "sd": 0.0,
            "n": 3,
            "display": "53.4 ± 0.0 (n=3)"
          },
          "Jev few-shot": {
            "mean": 55.3,
            "sd": 3.1,
            "n": 3,
            "display": "55.3 ± 3.1 (n=3)"
          }
        },
        "best": 71.8,
        "bestModels": [
          "SVM"
        ]
      },
      {
        "dataset": "Online Shoppers",
        "kind": "tabular",
        "testRows": 1000,
        "scores": {
          "Logistic regression": {
            "mean": 63.8,
            "sd": 11.0,
            "n": 3,
            "display": "63.8 ± 11.0 (n=3)"
          },
          "SVM": {
            "mean": 69.1,
            "sd": 0.3,
            "n": 3,
            "display": "69.1 ± 0.3 (n=3)"
          },
          "Decision tree": {
            "mean": 60.3,
            "sd": 5.1,
            "n": 3,
            "display": "60.3 ± 5.1 (n=3)"
          },
          "Random forest": {
            "mean": 56.8,
            "sd": 6.5,
            "n": 3,
            "display": "56.8 ± 6.5 (n=3)"
          },
          "Extra trees": {
            "mean": 53.3,
            "sd": 3.8,
            "n": 3,
            "display": "53.3 ± 3.8 (n=3)"
          },
          "k-NN": {
            "mean": 51.2,
            "sd": 1.3,
            "n": 3,
            "display": "51.2 ± 1.3 (n=3)"
          },
          "Naive Bayes": {
            "mean": 59.5,
            "sd": 0.4,
            "n": 3,
            "display": "59.5 ± 0.4 (n=3)"
          },
          "Hist gradient boost": {
            "mean": 51.8,
            "sd": 0.8,
            "n": 3,
            "display": "51.8 ± 0.8 (n=3)"
          },
          "XGBoost": {
            "mean": 65.1,
            "sd": 9.5,
            "n": 3,
            "display": "65.1 ± 9.5 (n=3)"
          },
          "CatBoost": {
            "mean": 63.7,
            "sd": 10.0,
            "n": 3,
            "display": "63.7 ± 10.0 (n=3)"
          },
          "Voting ensemble": {
            "mean": 63.5,
            "sd": 10.0,
            "n": 3,
            "display": "63.5 ± 10.0 (n=3)"
          },
          "Majority baseline": {
            "mean": 50.0,
            "sd": 0.0,
            "n": 3,
            "display": "50.0 ± 0.0 (n=3)"
          },
          "Jev zero-shot": {
            "mean": 51.4,
            "sd": 0.0,
            "n": 3,
            "display": "51.4 ± 0.0 (n=3)"
          },
          "Jev few-shot": {
            "mean": 54.7,
            "sd": 9.5,
            "n": 3,
            "display": "54.7 ± 9.5 (n=3)"
          }
        },
        "best": 69.1,
        "bestModels": [
          "SVM"
        ]
      },
      {
        "dataset": "Breast Cancer",
        "kind": "tabular",
        "testRows": 114,
        "scores": {
          "Logistic regression": {
            "mean": 99.5,
            "sd": 0.4,
            "n": 3,
            "display": "99.5 ± 0.4 (n=3)"
          },
          "SVM": {
            "mean": 100.0,
            "sd": 0.0,
            "n": 3,
            "display": "100.0 ± 0.0 (n=3)"
          },
          "Decision tree": {
            "mean": 94.0,
            "sd": 2.2,
            "n": 3,
            "display": "94.0 ± 2.2 (n=3)"
          },
          "Random forest": {
            "mean": 96.6,
            "sd": 1.9,
            "n": 3,
            "display": "96.6 ± 1.9 (n=3)"
          },
          "Extra trees": {
            "mean": 97.4,
            "sd": 1.2,
            "n": 3,
            "display": "97.4 ± 1.2 (n=3)"
          },
          "k-NN": {
            "mean": 100.0,
            "sd": 0.0,
            "n": 3,
            "display": "100.0 ± 0.0 (n=3)"
          },
          "Naive Bayes": {
            "mean": 92.2,
            "sd": 0.7,
            "n": 3,
            "display": "92.2 ± 0.7 (n=3)"
          },
          "Hist gradient boost": {
            "mean": 97.3,
            "sd": 0.6,
            "n": 3,
            "display": "97.3 ± 0.6 (n=3)"
          },
          "XGBoost": {
            "mean": 97.7,
            "sd": 0.4,
            "n": 3,
            "display": "97.7 ± 0.4 (n=3)"
          },
          "CatBoost": {
            "mean": 98.0,
            "sd": 0.6,
            "n": 3,
            "display": "98.0 ± 0.6 (n=3)"
          },
          "Voting ensemble": {
            "mean": 99.1,
            "sd": 0.4,
            "n": 3,
            "display": "99.1 ± 0.4 (n=3)"
          },
          "Majority baseline": {
            "mean": 50.0,
            "sd": 0.0,
            "n": 3,
            "display": "50.0 ± 0.0 (n=3)"
          },
          "Jev zero-shot": {
            "mean": 61.0,
            "sd": 0.0,
            "n": 3,
            "display": "61.0 ± 0.0 (n=3)"
          },
          "Jev few-shot": {
            "mean": 88.8,
            "sd": 5.5,
            "n": 3,
            "display": "88.8 ± 5.5 (n=3)"
          }
        },
        "best": 100.0,
        "bestModels": [
          "SVM",
          "k-NN"
        ]
      },
      {
        "dataset": "Iris",
        "kind": "tabular",
        "testRows": 30,
        "scores": {
          "Logistic regression": {
            "mean": 100.0,
            "sd": 0.0,
            "n": 3,
            "display": "100.0 ± 0.0 (n=3)"
          },
          "SVM": {
            "mean": 100.0,
            "sd": 0.0,
            "n": 3,
            "display": "100.0 ± 0.0 (n=3)"
          },
          "Decision tree": {
            "mean": 97.5,
            "sd": 2.1,
            "n": 3,
            "display": "97.5 ± 2.1 (n=3)"
          },
          "Random forest": {
            "mean": 98.9,
            "sd": 1.9,
            "n": 3,
            "display": "98.9 ± 1.9 (n=3)"
          },
          "Extra trees": {
            "mean": 100.0,
            "sd": 0.0,
            "n": 3,
            "display": "100.0 ± 0.0 (n=3)"
          },
          "k-NN": {
            "mean": 97.8,
            "sd": 3.9,
            "n": 3,
            "display": "97.8 ± 3.9 (n=3)"
          },
          "Naive Bayes": {
            "mean": 100.0,
            "sd": 0.0,
            "n": 3,
            "display": "100.0 ± 0.0 (n=3)"
          },
          "Hist gradient boost": {
            "mean": 100.0,
            "sd": 0.0,
            "n": 3,
            "display": "100.0 ± 0.0 (n=3)"
          },
          "XGBoost": {
            "mean": 95.7,
            "sd": 2.1,
            "n": 3,
            "display": "95.7 ± 2.1 (n=3)"
          },
          "CatBoost": {
            "mean": 97.5,
            "sd": 2.1,
            "n": 3,
            "display": "97.5 ± 2.1 (n=3)"
          },
          "Voting ensemble": {
            "mean": 100.0,
            "sd": 0.0,
            "n": 3,
            "display": "100.0 ± 0.0 (n=3)"
          },
          "Majority baseline": {
            "mean": 33.3,
            "sd": 0.0,
            "n": 3,
            "display": "33.3 ± 0.0 (n=3)"
          },
          "Jev zero-shot": {
            "mean": 97.0,
            "sd": 0.0,
            "n": 3,
            "display": "97.0 ± 0.0 (n=3)"
          },
          "Jev few-shot": {
            "mean": 94.5,
            "sd": 4.8,
            "n": 3,
            "display": "94.5 ± 4.8 (n=3)"
          }
        },
        "best": 100.0,
        "bestModels": [
          "Logistic regression",
          "SVM",
          "Extra trees",
          "Naive Bayes",
          "Hist gradient boost",
          "Voting ensemble"
        ]
      }
    ]
  },
  "adjusted": {
    "models": [
      "Logistic regression",
      "SVM",
      "Decision tree",
      "Random forest",
      "Extra trees",
      "k-NN",
      "Naive Bayes",
      "Hist gradient boost",
      "XGBoost",
      "CatBoost",
      "Voting ensemble",
      "Majority baseline",
      "Jev zero-shot",
      "Jev few-shot"
    ],
    "rows": [
      {
        "dataset": "AG News",
        "kind": "text",
        "testRows": 1000,
        "scores": {
          "Logistic regression": {
            "mean": 87.4,
            "sd": 0.8,
            "n": 3,
            "display": "87.4 ± 0.8 (n=3)"
          },
          "SVM": {
            "mean": 88.4,
            "sd": 0.3,
            "n": 3,
            "display": "88.4 ± 0.3 (n=3)"
          },
          "Decision tree": {
            "mean": 67.5,
            "sd": 0.9,
            "n": 3,
            "display": "67.5 ± 0.9 (n=3)"
          },
          "Random forest": {
            "mean": 71.6,
            "sd": 1.1,
            "n": 3,
            "display": "71.6 ± 1.1 (n=3)"
          },
          "Extra trees": {
            "mean": 75.1,
            "sd": 0.7,
            "n": 3,
            "display": "75.1 ± 0.7 (n=3)"
          },
          "k-NN": {
            "mean": 78.4,
            "sd": 0.3,
            "n": 3,
            "display": "78.4 ± 0.3 (n=3)"
          },
          "Naive Bayes": {
            "mean": 87.4,
            "sd": 0.2,
            "n": 3,
            "display": "87.4 ± 0.2 (n=3)"
          },
          "Hist gradient boost": {
            "mean": 80.3,
            "sd": 0.8,
            "n": 3,
            "display": "80.3 ± 0.8 (n=3)"
          },
          "XGBoost": {
            "mean": 82.6,
            "sd": 0.2,
            "n": 3,
            "display": "82.6 ± 0.2 (n=3)"
          },
          "CatBoost": {
            "mean": 82.0,
            "sd": 1.0,
            "n": 3,
            "display": "82.0 ± 1.0 (n=3)"
          },
          "Voting ensemble": {
            "mean": 87.0,
            "sd": 1.0,
            "n": 3,
            "display": "87.0 ± 1.0 (n=3)"
          },
          "Majority baseline": {
            "mean": 25.0,
            "sd": 0.0,
            "n": 3,
            "display": "25.0 ± 0.0 (n=3)"
          },
          "Jev zero-shot": {
            "mean": 87.5,
            "sd": 0.0,
            "n": 3,
            "display": "87.5 ± 0.0 (n=3)"
          },
          "Jev few-shot": {
            "mean": 86.3,
            "sd": 0.6,
            "n": 3,
            "display": "86.3 ± 0.6 (n=3)"
          }
        },
        "best": 88.4,
        "bestModels": [
          "SVM"
        ]
      },
      {
        "dataset": "Banking77",
        "kind": "text",
        "testRows": 1500,
        "scores": {
          "Logistic regression": {
            "mean": 89.4,
            "sd": 0.2,
            "n": 3,
            "display": "89.4 ± 0.2 (n=3)"
          },
          "SVM": {
            "mean": 89.7,
            "sd": 0.6,
            "n": 3,
            "display": "89.7 ± 0.6 (n=3)"
          },
          "Decision tree": {
            "mean": 62.2,
            "sd": 0.9,
            "n": 3,
            "display": "62.2 ± 0.9 (n=3)"
          },
          "Random forest": {
            "mean": 68.2,
            "sd": 1.4,
            "n": 3,
            "display": "68.2 ± 1.4 (n=3)"
          },
          "Extra trees": {
            "mean": 69.8,
            "sd": 0.4,
            "n": 3,
            "display": "69.8 ± 0.4 (n=3)"
          },
          "k-NN": {
            "mean": 58.4,
            "sd": 1.9,
            "n": 3,
            "display": "58.4 ± 1.9 (n=3)"
          },
          "Naive Bayes": {
            "mean": 86.0,
            "sd": 0.5,
            "n": 3,
            "display": "86.0 ± 0.5 (n=3)"
          },
          "Hist gradient boost": {
            "mean": 62.2,
            "sd": 1.2,
            "n": 3,
            "display": "62.2 ± 1.2 (n=3)"
          },
          "XGBoost": {
            "mean": 71.9,
            "sd": 0.4,
            "n": 3,
            "display": "71.9 ± 0.4 (n=3)"
          },
          "CatBoost": {
            "mean": 68.1,
            "sd": 0.7,
            "n": 3,
            "display": "68.1 ± 0.7 (n=3)"
          },
          "Voting ensemble": {
            "mean": 86.2,
            "sd": 0.4,
            "n": 3,
            "display": "86.2 ± 0.4 (n=3)"
          },
          "Majority baseline": {
            "mean": 1.3,
            "sd": 0.0,
            "n": 3,
            "display": "1.3 ± 0.0 (n=3)"
          },
          "Jev zero-shot": {
            "mean": 78.9,
            "sd": 0.0,
            "n": 3,
            "display": "78.9 ± 0.0 (n=3)"
          },
          "Jev few-shot": {
            "mean": 81.9,
            "sd": 1.7,
            "n": 3,
            "display": "81.9 ± 1.7 (n=3)"
          }
        },
        "best": 89.7,
        "bestModels": [
          "SVM"
        ]
      },
      {
        "dataset": "SMS Spam",
        "kind": "text",
        "testRows": 1000,
        "scores": {
          "Logistic regression": {
            "mean": 95.6,
            "sd": 0.7,
            "n": 3,
            "display": "95.6 ± 0.7 (n=3)"
          },
          "SVM": {
            "mean": 96.2,
            "sd": 0.8,
            "n": 3,
            "display": "96.2 ± 0.8 (n=3)"
          },
          "Decision tree": {
            "mean": 88.5,
            "sd": 0.6,
            "n": 3,
            "display": "88.5 ± 0.6 (n=3)"
          },
          "Random forest": {
            "mean": 96.2,
            "sd": 0.9,
            "n": 3,
            "display": "96.2 ± 0.9 (n=3)"
          },
          "Extra trees": {
            "mean": 94.5,
            "sd": 0.8,
            "n": 3,
            "display": "94.5 ± 0.8 (n=3)"
          },
          "k-NN": {
            "mean": 92.4,
            "sd": 0.8,
            "n": 3,
            "display": "92.4 ± 0.8 (n=3)"
          },
          "Naive Bayes": {
            "mean": 96.3,
            "sd": 0.6,
            "n": 3,
            "display": "96.3 ± 0.6 (n=3)"
          },
          "Hist gradient boost": {
            "mean": 93.8,
            "sd": 0.3,
            "n": 3,
            "display": "93.8 ± 0.3 (n=3)"
          },
          "XGBoost": {
            "mean": 90.7,
            "sd": 1.2,
            "n": 3,
            "display": "90.7 ± 1.2 (n=3)"
          },
          "CatBoost": {
            "mean": 94.1,
            "sd": 0.6,
            "n": 3,
            "display": "94.1 ± 0.6 (n=3)"
          },
          "Voting ensemble": {
            "mean": 95.3,
            "sd": 1.1,
            "n": 3,
            "display": "95.3 ± 1.1 (n=3)"
          },
          "Majority baseline": {
            "mean": 50.0,
            "sd": 0.0,
            "n": 3,
            "display": "50.0 ± 0.0 (n=3)"
          },
          "Jev zero-shot": {
            "mean": 95.9,
            "sd": 0.7,
            "n": 3,
            "display": "95.9 ± 0.7 (n=3)"
          },
          "Jev few-shot": {
            "mean": 95.8,
            "sd": 0.5,
            "n": 3,
            "display": "95.8 ± 0.5 (n=3)"
          }
        },
        "best": 96.3,
        "bestModels": [
          "Naive Bayes"
        ]
      },
      {
        "dataset": "IMDb",
        "kind": "text",
        "testRows": 1000,
        "scores": {
          "Logistic regression": {
            "mean": 88.3,
            "sd": 0.2,
            "n": 3,
            "display": "88.3 ± 0.2 (n=3)"
          },
          "SVM": {
            "mean": 87.0,
            "sd": 1.9,
            "n": 3,
            "display": "87.0 ± 1.9 (n=3)"
          },
          "Decision tree": {
            "mean": 70.5,
            "sd": 0.6,
            "n": 3,
            "display": "70.5 ± 0.6 (n=3)"
          },
          "Random forest": {
            "mean": 81.3,
            "sd": 0.7,
            "n": 3,
            "display": "81.3 ± 0.7 (n=3)"
          },
          "Extra trees": {
            "mean": 83.0,
            "sd": 1.5,
            "n": 3,
            "display": "83.0 ± 1.5 (n=3)"
          },
          "k-NN": {
            "mean": 80.6,
            "sd": 1.5,
            "n": 3,
            "display": "80.6 ± 1.5 (n=3)"
          },
          "Naive Bayes": {
            "mean": 85.3,
            "sd": 0.9,
            "n": 3,
            "display": "85.3 ± 0.9 (n=3)"
          },
          "Hist gradient boost": {
            "mean": 83.1,
            "sd": 0.2,
            "n": 3,
            "display": "83.1 ± 0.2 (n=3)"
          },
          "XGBoost": {
            "mean": 83.7,
            "sd": 0.2,
            "n": 3,
            "display": "83.7 ± 0.2 (n=3)"
          },
          "CatBoost": {
            "mean": 83.6,
            "sd": 0.5,
            "n": 3,
            "display": "83.6 ± 0.5 (n=3)"
          },
          "Voting ensemble": {
            "mean": 86.8,
            "sd": 0.3,
            "n": 3,
            "display": "86.8 ± 0.3 (n=3)"
          },
          "Majority baseline": {
            "mean": 50.0,
            "sd": 0.0,
            "n": 3,
            "display": "50.0 ± 0.0 (n=3)"
          },
          "Jev zero-shot": {
            "mean": 96.1,
            "sd": 0.6,
            "n": 3,
            "display": "96.1 ± 0.6 (n=3)"
          },
          "Jev few-shot": {
            "mean": 95.5,
            "sd": 0.8,
            "n": 3,
            "display": "95.5 ± 0.8 (n=3)"
          }
        },
        "best": 88.3,
        "bestModels": [
          "Logistic regression"
        ]
      },
      {
        "dataset": "Bank Marketing",
        "kind": "tabular",
        "testRows": 1000,
        "scores": {
          "Logistic regression": {
            "mean": 68.9,
            "sd": 2.4,
            "n": 3,
            "display": "68.9 ± 2.4 (n=3)"
          },
          "SVM": {
            "mean": 71.4,
            "sd": 2.0,
            "n": 3,
            "display": "71.4 ± 2.0 (n=3)"
          },
          "Decision tree": {
            "mean": 68.2,
            "sd": 2.5,
            "n": 3,
            "display": "68.2 ± 2.5 (n=3)"
          },
          "Random forest": {
            "mean": 73.0,
            "sd": 1.2,
            "n": 3,
            "display": "73.0 ± 1.2 (n=3)"
          },
          "Extra trees": {
            "mean": 72.1,
            "sd": 0.1,
            "n": 3,
            "display": "72.1 ± 0.1 (n=3)"
          },
          "k-NN": {
            "mean": 66.9,
            "sd": 0.4,
            "n": 3,
            "display": "66.9 ± 0.4 (n=3)"
          },
          "Naive Bayes": {
            "mean": 64.9,
            "sd": 2.6,
            "n": 3,
            "display": "64.9 ± 2.6 (n=3)"
          },
          "Hist gradient boost": {
            "mean": 72.4,
            "sd": 1.2,
            "n": 3,
            "display": "72.4 ± 1.2 (n=3)"
          },
          "XGBoost": {
            "mean": 71.4,
            "sd": 2.5,
            "n": 3,
            "display": "71.4 ± 2.5 (n=3)"
          },
          "CatBoost": {
            "mean": 70.1,
            "sd": 2.8,
            "n": 3,
            "display": "70.1 ± 2.8 (n=3)"
          },
          "Voting ensemble": {
            "mean": 73.3,
            "sd": 0.3,
            "n": 3,
            "display": "73.3 ± 0.3 (n=3)"
          },
          "Majority baseline": {
            "mean": 50.0,
            "sd": 0.0,
            "n": 3,
            "display": "50.0 ± 0.0 (n=3)"
          },
          "Jev zero-shot": {
            "mean": 59.7,
            "sd": 1.6,
            "n": 3,
            "display": "59.7 ± 1.6 (n=3)"
          },
          "Jev few-shot": {
            "mean": 59.0,
            "sd": 2.5,
            "n": 3,
            "display": "59.0 ± 2.5 (n=3)"
          }
        },
        "best": 73.3,
        "bestModels": [
          "Voting ensemble"
        ]
      },
      {
        "dataset": "Online Shoppers",
        "kind": "tabular",
        "testRows": 1000,
        "scores": {
          "Logistic regression": {
            "mean": 69.3,
            "sd": 1.1,
            "n": 3,
            "display": "69.3 ± 1.1 (n=3)"
          },
          "SVM": {
            "mean": 68.4,
            "sd": 0.4,
            "n": 3,
            "display": "68.4 ± 0.4 (n=3)"
          },
          "Decision tree": {
            "mean": 63.6,
            "sd": 0.2,
            "n": 3,
            "display": "63.6 ± 0.2 (n=3)"
          },
          "Random forest": {
            "mean": 69.7,
            "sd": 0.6,
            "n": 3,
            "display": "69.7 ± 0.6 (n=3)"
          },
          "Extra trees": {
            "mean": 68.4,
            "sd": 1.2,
            "n": 3,
            "display": "68.4 ± 1.2 (n=3)"
          },
          "k-NN": {
            "mean": 65.9,
            "sd": 1.5,
            "n": 3,
            "display": "65.9 ± 1.5 (n=3)"
          },
          "Naive Bayes": {
            "mean": 64.5,
            "sd": 2.3,
            "n": 3,
            "display": "64.5 ± 2.3 (n=3)"
          },
          "Hist gradient boost": {
            "mean": 69.2,
            "sd": 3.5,
            "n": 3,
            "display": "69.2 ± 3.5 (n=3)"
          },
          "XGBoost": {
            "mean": 69.5,
            "sd": 1.4,
            "n": 3,
            "display": "69.5 ± 1.4 (n=3)"
          },
          "CatBoost": {
            "mean": 69.8,
            "sd": 1.9,
            "n": 3,
            "display": "69.8 ± 1.9 (n=3)"
          },
          "Voting ensemble": {
            "mean": 71.2,
            "sd": 1.4,
            "n": 3,
            "display": "71.2 ± 1.4 (n=3)"
          },
          "Majority baseline": {
            "mean": 50.0,
            "sd": 0.0,
            "n": 3,
            "display": "50.0 ± 0.0 (n=3)"
          },
          "Jev zero-shot": {
            "mean": 51.8,
            "sd": 0.1,
            "n": 3,
            "display": "51.8 ± 0.1 (n=3)"
          },
          "Jev few-shot": {
            "mean": 53.6,
            "sd": 7.3,
            "n": 3,
            "display": "53.6 ± 7.3 (n=3)"
          }
        },
        "best": 71.2,
        "bestModels": [
          "Voting ensemble"
        ]
      },
      {
        "dataset": "Breast Cancer",
        "kind": "tabular",
        "testRows": 114,
        "scores": {
          "Logistic regression": {
            "mean": 100.0,
            "sd": 0.0,
            "n": 3,
            "display": "100.0 ± 0.0 (n=3)"
          },
          "SVM": {
            "mean": 99.6,
            "sd": 0.7,
            "n": 3,
            "display": "99.6 ± 0.7 (n=3)"
          },
          "Decision tree": {
            "mean": 93.5,
            "sd": 1.4,
            "n": 3,
            "display": "93.5 ± 1.4 (n=3)"
          },
          "Random forest": {
            "mean": 96.9,
            "sd": 1.5,
            "n": 3,
            "display": "96.9 ± 1.5 (n=3)"
          },
          "Extra trees": {
            "mean": 97.1,
            "sd": 0.5,
            "n": 3,
            "display": "97.1 ± 0.5 (n=3)"
          },
          "k-NN": {
            "mean": 99.5,
            "sd": 0.8,
            "n": 3,
            "display": "99.5 ± 0.8 (n=3)"
          },
          "Naive Bayes": {
            "mean": 93.3,
            "sd": 1.6,
            "n": 3,
            "display": "93.3 ± 1.6 (n=3)"
          },
          "Hist gradient boost": {
            "mean": 96.0,
            "sd": 2.5,
            "n": 3,
            "display": "96.0 ± 2.5 (n=3)"
          },
          "XGBoost": {
            "mean": 98.0,
            "sd": 0.5,
            "n": 3,
            "display": "98.0 ± 0.5 (n=3)"
          },
          "CatBoost": {
            "mean": 97.9,
            "sd": 0.7,
            "n": 3,
            "display": "97.9 ± 0.7 (n=3)"
          },
          "Voting ensemble": {
            "mean": 98.7,
            "sd": 0.7,
            "n": 3,
            "display": "98.7 ± 0.7 (n=3)"
          },
          "Majority baseline": {
            "mean": 50.0,
            "sd": 0.0,
            "n": 3,
            "display": "50.0 ± 0.0 (n=3)"
          },
          "Jev zero-shot": {
            "mean": 88.4,
            "sd": 2.2,
            "n": 3,
            "display": "88.4 ± 2.2 (n=3)"
          },
          "Jev few-shot": {
            "mean": 92.7,
            "sd": 0.9,
            "n": 3,
            "display": "92.7 ± 0.9 (n=3)"
          }
        },
        "best": 100.0,
        "bestModels": [
          "Logistic regression"
        ]
      },
      {
        "dataset": "Iris",
        "kind": "tabular",
        "testRows": 30,
        "scores": {
          "Logistic regression": {
            "mean": 100.0,
            "sd": 0.0,
            "n": 3,
            "display": "100.0 ± 0.0 (n=3)"
          },
          "SVM": {
            "mean": 100.0,
            "sd": 0.0,
            "n": 3,
            "display": "100.0 ± 0.0 (n=3)"
          },
          "Decision tree": {
            "mean": 97.5,
            "sd": 2.1,
            "n": 3,
            "display": "97.5 ± 2.1 (n=3)"
          },
          "Random forest": {
            "mean": 98.9,
            "sd": 1.9,
            "n": 3,
            "display": "98.9 ± 1.9 (n=3)"
          },
          "Extra trees": {
            "mean": 100.0,
            "sd": 0.0,
            "n": 3,
            "display": "100.0 ± 0.0 (n=3)"
          },
          "k-NN": {
            "mean": 97.8,
            "sd": 3.9,
            "n": 3,
            "display": "97.8 ± 3.9 (n=3)"
          },
          "Naive Bayes": {
            "mean": 100.0,
            "sd": 0.0,
            "n": 3,
            "display": "100.0 ± 0.0 (n=3)"
          },
          "Hist gradient boost": {
            "mean": 100.0,
            "sd": 0.0,
            "n": 3,
            "display": "100.0 ± 0.0 (n=3)"
          },
          "XGBoost": {
            "mean": 95.7,
            "sd": 2.1,
            "n": 3,
            "display": "95.7 ± 2.1 (n=3)"
          },
          "CatBoost": {
            "mean": 97.5,
            "sd": 2.1,
            "n": 3,
            "display": "97.5 ± 2.1 (n=3)"
          },
          "Voting ensemble": {
            "mean": 100.0,
            "sd": 0.0,
            "n": 3,
            "display": "100.0 ± 0.0 (n=3)"
          },
          "Majority baseline": {
            "mean": 33.3,
            "sd": 0.0,
            "n": 3,
            "display": "33.3 ± 0.0 (n=3)"
          },
          "Jev zero-shot": {
            "mean": 97.0,
            "sd": 0.0,
            "n": 3,
            "display": "97.0 ± 0.0 (n=3)"
          },
          "Jev few-shot": {
            "mean": 94.5,
            "sd": 4.8,
            "n": 3,
            "display": "94.5 ± 4.8 (n=3)"
          }
        },
        "best": 100.0,
        "bestModels": [
          "Logistic regression",
          "SVM",
          "Extra trees",
          "Naive Bayes",
          "Hist gradient boost",
          "Voting ensemble"
        ]
      }
    ]
  }
};
