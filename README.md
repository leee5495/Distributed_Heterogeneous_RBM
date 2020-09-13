# Distributed Heterogeneous RBM
Experimentation code for book recommendation with Distributed Heterogeneous RBM

![image](https://github.com/leee5495/Distributed_Heterogeneous_RBM/blob/master/misc/%EB%8F%84%ED%98%95.png)

### Data
ratings data available [here](https://drive.google.com/file/d/1nR7B7fDzwwExYpO70xY6FYTGf3aVXX_T/view?usp=sharing)
(email leee5495@gmail.com for access)

### train.py
trains Distributed Heterogeneous RBM and saves the model
- `datapath` = directory path to the train data
- `modelpath` = directory path to save the trained model
- hyperparameters
  ```
  k = number of contrastive divergence steps
  num_cluster = number of cluster rbms to use
  batch_size = batch size for training
  epochs = number of epochs for base rbm train
  bootstrap_epochs = number of additional epochs to train each cluster rbm
  ensemble_epochs = number of epochs to train the ensemble model
  ```
- run `test.explain_prediction(output_vec, input_vec)` to view recommendation result
