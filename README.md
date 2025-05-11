#  Text Classification with RNN & LSTM

##  Overview
This project implements and compares Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) models for sentiment classification on the IMDB movie reviews dataset. It includes:
- Data preprocessing (tokenization, vocabulary building, sequence padding)
- RNN and LSTM models (both built-in and from scratch)
- Training and evaluation pipelines
- Sentiment prediction on new reviews

##  Project Structure
`notebooks/` - Jupyter notebook with all training & evaluation\
`requirements.txt` - Dependencies



## Models Implemented
- **RNN (PyTorch Built-in)**
- **Custom RNN (Implemented from Scratch)**
- **LSTM (PyTorch Built-in)**
- **Custom LSTM (Implemented from Scratch)**

##  Results
| Model  | Accuracy on Test Set |
|--------|----------------------|
| **Built-in RNN** | 58% |
| **Custom RNN** | 64% |
| **Built-in LSTM** | 90% |
| **Custom LSTM** | 92% |

### Classification Reports
Built-in RNN Model:
```yaml
                precision    recall  f1-score   support

    Negative       0.59      0.68      0.63      1619
    Positive       0.58      0.48      0.52      1488

    accuracy                           0.58      3107
   macro avg       0.58      0.58      0.58      3107
weighted avg       0.58      0.58      0.58      3107
```
Custom RNN Model:
```yaml
                precision    recall  f1-score   support

    Negative       0.62      0.81      0.70      1619
    Positive       0.68      0.46      0.55      1488

    accuracy                           0.64      3107
   macro avg       0.65      0.63      0.62      3107
weighted avg       0.65      0.64      0.63      3107

```
Built-in LSTM Model:
```yaml
                precision    recall  f1-score   support

    Negative       0.90      0.91      0.90      1619
    Positive       0.90      0.88      0.89      1488

    accuracy                           0.90      3107
   macro avg       0.90      0.90      0.90      3107
weighted avg       0.90      0.90      0.90      3107

```
Custom LSTM Model:
```yaml
                precision    recall  f1-score   support

    Negative       0.92      0.93      0.93      1619
    Positive       0.92      0.91      0.92      1488

    accuracy                           0.92      3107
   macro avg       0.92      0.92      0.92      3107
weighted avg       0.92      0.92      0.92      3107
```
### Key Takeaways
- RNNs struggle with long sequences due to the vanishing gradient problem.
- LSTMs capture long-term dependencies better, leading to higher accuracy.
- Hyperparameters such as hidden size and learning rate significantly impact model performance.


##  References

- Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. [Neural Computation](https://www.bioinf.jku.at/publications/older/2604.pdf)
- Mikolov, T., Karafiát, M., Burget, L., Černocký, J., & Khudanpur, S. (2010). *Recurrent Neural Network based Language Model*. [Interspeech 2010](https://www.isca-speech.org/archive/archive_papers/interspeech_2010/i10_1045.pdf)
- Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). *Learning Word Vectors for Sentiment Analysis*. [ACL 2011](https://aclanthology.org/P11-1015/)
- PyTorch Documentation: [RNN module](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html), [LSTM module](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- IMDB Dataset: [https://ai.stanford.edu/~amaas/data/sentiment/](https://ai.stanford.edu/~amaas/data/sentiment/)
