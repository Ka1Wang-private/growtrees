# Cost-Aware Robust Tree Ensembles for Security Applications
Code for the paper "[Cost-Aware Robust Tree Ensembles for Security Applications](https://arxiv.org/abs/1912.01149)" (USENIX Security'21), Yizheng Chen, Shiqi Wang, Weifan Jiang, Asaf Cidon, Suman Jana.

Blog Post: https://surrealyz.medium.com/robust-trees-for-security-577061177320

We utilize security domain knowledge to increase the evasion cost against security classifiers, specifically, tree ensemble models that are widely used by security tasks. We propose a new cost modeling method to capture the domain knowledge of features as constraint, and then we integrate the cost-driven constraint into the node construction process to train robust tree ensembles. During the training process, we use the constraint to find data points that are likely to be perturbed given the costs of the features, and we optimize the quality of the trees using a new robust training algorithm. Our cost-aware training method can be applied to different types of tree ensembles, including random forest model (scikit-learn) and gradient boosted decision trees (XGBoost).

## Robust training algorithm

### Implementation in scikit-learn

* Clone our dev version of [scikit-learn](https://github.com/surrealyz/scikit-learn/)
* Check out the [robust](https://github.com/surrealyz/scikit-learn/tree/robust) branch
* We recommend using a virtualenv to install this
* After activating your virtualenv, install the required packages ```pip install numpy scipy joblib threadpoolctl Cython```
* Then install sklearn with our robust training algorithm ```python setup.py install```
* Run `data/download_data.sh` under the current repo [(source)](https://github.com/chenhongge/RobustTrees/blob/master/data/download_data.sh)
* Example usage
  ```
  python train_rf_one.py --train data/binary_mnist0
                        --test data/binary_mnist0.t
                        -m models/rf/greedy/sklearn_greedy_binary_mnist.pickle
                        -b -z -n 784 -r -s robust -e 0.3
                        -c gini --nt 1000 -d 6
  ```

### Implementation in XGBoost

* Clone our dev version of [XGBoost RobustTrees](https://github.com/surrealyz/RobustTrees)
* Check out the [greedy](https://github.com/surrealyz/RobustTrees/tree/greedy) branch
* Run `build.sh`
* `gunzip` all the `*.csv.gz` files under `RobustTrees/data` to obtain the csv datasets. Reading libsvm sometimes has issues in that version of XGBoost, so we converted the dataset to csv files.
* Example usage
  ```
  ./xgboost data/breast_cancer.greedy.conf
  ```

## Datasets

We evaluated our core training algorithm without cost constraints over four benchmark datasets, see the table below.

| Dataset | Train set size  | Test set size  | Majority class in train, test (%)  | # of features  |
|---|---|---|---|---|
| breast-cancer  | 546 | 137  | 62.64, 74.45  | 10  |
| cod-rna  | 59,535  | 271,617  | 66.67, 66.67  | 8  |
| ijcnn1  | 49,990  | 91,701  | 90.29, 90.50  | 22  |
| MNIST 2 vs. 6  | 11,876  | 1,990  | 50.17, 51.86  | 784  |

We have also evaluated our cost-aware training algorihtm over a Twitter spam detection dataset used in the paper ["A Domain-Agnostic Approach to Spam-URL Detection via Redirects"](https://www.andrew.cmu.edu/user/lakoglu/pubs/17-pakdd-urlspam.pdf). We re-extracted 25 features (see Table 7 in our paper) as the Twitter spam detection dataset.

| Twitter spam dataset  | Training  |  Testing |
|---|---|---|
| Malicious  | 130,794  | 55,732  |
| Benign  | 165,076  | 71,070  |
| Total  | 295,870  | 126,802  |

Both datasets are available in `data/`, and the files need to be uncompressed.
Please also run `cd data/; ./download_data.sh` to get libsvm files under `data/` directory, since some of our Python scripts read the libsvm data.

## Benchmark datasets evaluation

### GBDT models

#### Trained models in the paper

* Regular training, **natural** model in the paper: `models/gbdt/nature_*.bin`
* [Chen's robust training algorithm](https://github.com/chenhongge/RobustTrees), **Chen's** model in the paper: `models/gbdt/robust_*.bin`
* Our training algorithm, **ours** model in the paper: `models/gbdt/greedy_*.bin`

#### Evaluate the models

* **Performance:** To evaluate model accuracy, false positive rate, AUC, and plot the ROC curves, please run the following commands:
  * `python scripts/xgboost_roc_plots.py breast_cancer`
  * `python scripts/xgboost_roc_plots.py ijcnn`
  * `python scripts/xgboost_roc_plots.py cod-rna`
  * `python scripts/xgboost_roc_plots.py binary_mnist`
  * The model performance numbers correspond to Table 3, and the generated plots in `roc_plots/` correspond to Figure 7 in the paper.
* **Robustness:** To evaluate the robustness of models, we use the MILP attack: `xgbKantchelianAttack.py`. It uses Gurobi solver, so you need to obtain a licence from Gurobi to use it. They provide free academic license.
  * `mkdir logs`
  * `mkdir -p adv_examples/gbdt`
  * `mkdir -p result/gbdt`
  * breast_cancer:
    ```
    for mtype in $(echo 'nature' 'robust' 'greedy'); do dt='breast_cancer'; python xgbKantchelianAttack.py --data 'data/breast_cancer_scale0.test' --model_type 'xgboost' --model "models/gbdt/${mtype}_${dt}.bin" --rand --num_classes 2 --nfeat 10 --feature_start 1 --both --maxone -n 100 --out "result/gbdt/${mtype}_${dt}.txt" --adv "adv_examples/gbdt/${mtype}_${dt}_adv.pickle" >! logs/milp_gbdt_${mtype}_${dt}.log 2>&1&; done
    ```
  * cod-rna
    ```
    for md in $(echo 'nature_cod-rna' 'robust_cod-rna' 'greedy_cod-rna_center_eps0.03'); do python xgbKantchelianAttack.py --data 'data/cod-rna_s.t' --model_type 'xgboost' --model "models/gbdt/${md}.bin" --rand --num_classes 2 --nfeat 8 --feature_start 0 --both --maxone -n 5000 --out "result/gbdt/${md}.txt" --adv "adv_examples/gbdt/${md}_adv.pickle" >! logs/milp_gbdt_${md}.log 2>&1&; done
    ```
  * ijcnn:
    ```
    for md in $(echo 'nature_ijcnn' 'robust_ijcnn' 'greedy_ijcnn_center_eps0.02_nr60_md8'); do python xgbKantchelianAttack.py --data 'data/ijcnn1s0.t' --model_type 'xgboost' --model "models/gbdt/${md}.bin" --rand --num_classes 2 --nfeat 22 --feature_start 1 --both --maxone -n 100 --out "result/gbdt/${md}.txt" --adv "adv_examples/gbdt/${md}_adv.pickle" >! logs/milp_gbdt_${md}.log 2>&1&; done
    ```
  * binary_mnist:
    ```
    for md in $(echo 'nature_binary_mnist' 'robust_binary_mnist' 'greedy_binary_mnist'); do python xgbKantchelianAttack.py -n 100 --data 'data/binary_mnist_round6.test.csv' --model_type 'xgboost' --model "models/gbdt/${md}.bin" --rand --num_classes 2 --nfeat 784 --both --maxone --feature_start 0 --out "result/gbdt/${md}.txt" --adv "adv_examples/gbdt/${md}.pickle" >! logs/milp_gbdt_${md}.log 2>&1&; done
    ```

#### How to train the models

In the cloned [greedy branch of RobustTrees repo](https://github.com/surrealyz/RobustTrees/tree/greedy), after building the `xgboost` binary file, the following commands train the **natural**, **Chen's**, and **Ours** models respectively:
```
./xgboost data/breast_cancer.unrob.conf
./xgboost data/breast_cancer.conf
./xgboost data/breast_cancer.greedy.conf
```

For cod-rna, ijcnn, and binary_mnist datasets, the commands follow the same style, `./xgboost data/${dataset}.unrob.conf`, `./xgboost data/${dataset}.conf`, and `./xgboost data/${dataset}.greedy.conf`.


### Random Forest models

#### Trained models in the paper

* Regular training, **natural** model in the paper: `models/rf/*best*.bin`
* [Chen's robust training algorithm](https://github.com/chenhongge/RobustTrees), **Chen's** model in the paper: `models/gbdt/*heuristic*.bin`
* Our training algorithm, **ours** model in the paper: `models/gbdt/*robust*.bin`

#### Evaluate the models

* **Performance:** To evaluate model accuracy, false positive rate, AUC, and plot the roc curve figures, please run the following commands:
  * `python scripts/sklearn_roc_scripts.py breast_cancer`
  * `python scripts/sklearn_roc_scripts.py ijcnn`
  * `python scripts/sklearn_roc_scripts.py cod-rna`
  * `python scripts/sklearn_roc_scripts.py binary_mnist`

  * The model performance numbers correspond to Table 4, and the generated plots in `roc_plots/` correspond to Figure 9 in the paper.

* **Robustness:** To evaluate the robustness of models, we use the MILP attack: `xgbKantchelianAttack.py`. It uses Gurobi solver, so you need to obtain a licence from Gurobi to use it. They provide free academic license.
  * `mkdir logs`
  * `mkdir -p result/sk-rf`
  * use `attack_sklearn_selected_RF.py`

#### How to train the models

The script `train_all_sklearn.py` trains all models, where the splitter choice `best` is **natural**, `heuristic` is **Chen's**, and `robust` is **ours**. You can modify the loop or use `train_rf_one.py` to train an individual model.

## Twitter Spam Detection Application

### Trained models in the paper

`models/gbdt/twitter/`

### How to train the models

For example, to train model M19, run this in RobustTrees: `./xgboost data/twitter_spam.greedy.flex.conf`

### Evaluate the models

* **Performance:** 
`scripts/model_accuracy.py`

* **Robustness:**
The following are examples to run the six attacks against the `twitter_spam_nature` model. Change the model name accordingly for other models.
  * l_1:
    ```
    md='twitter_spam_nature'; r='_l1'; python xgbKantchelianAttack.py -n 100 --order 1 --data 'data/500_malicious.libsvm' --model_type 'xgboost' --model "models/twitter/${md}.bin" --num_classes 2 --nfeat 25 --maxone --feature_start 0 --out "result/gbdt/adap_${md}${r}.txt" --adv "adv_examples/gbdt/${md}${r}.pickle" >! logs/milp_gbdt_adap_${md}${r}.log &
    ```
  * l_2:
    ```
    md='twitter_spam_nature'; r='_l2'; python xgbKantchelianAttack.py -n 100 --order 2 --data 'data/500_malicious.libsvm' --model_type 'xgboost' --model "models/twitter/${md}.bin" --num_classes 2 --nfeat 25 --maxone --feature_start 0 --out "result/gbdt/adap_${md}${r}.txt" --adv "adv_examples/gbdt/${md}${r}.pickle" >! logs/milp_gbdt_adap_${md}${r}.log &
    ```
  * cost_1:
    ```
    o='obj1';b='bound_obj1'; md='twitter_spam_nature'; python flexible_xgbKantchelianAttack_cost.py -n 100 --weight config/weight/${o}.json -b "config/eps/${b}.json" --data 'data/500_malicious.libsvm' --model_type 'xgboost' --model "models/twitter/${md}.bin" --num_classes 2 --feature_start 0 --out "result/gbdt/adap_${md}_${o}.txt" --adv "adv_examples/gbdt/${md}_${o}.pickle" >! logs/milp_gbdt_adap_${md}_${o}.log &
    ```
  * cost_2:
    ```
    o='obj2'; md='twitter_spam_nature'; python flexible_xgbKantchelianAttack_cost.py -n 100 --weight "config/weight/${o}.json" --data 'data/500_malicious.libsvm' --model_type 'xgboost' --model "models/twitter/${md}.bin" --num_classes 2 --feature_start 0 --out "result/gbdt/adap_${md}_${o}.txt" --adv "adv_examples/gbdt/${md}_${o}.pickle" >! logs/milp_gbdt_adap_${md}_${o}.log &
    ```
  * cost_3:
    ```
    o='obj3';b='bound_obj3'; md='twitter_spam_nature'; python flexible_xgbKantchelianAttack_cost.py -n 100 --weight config/weight/${o}.json -b "config/eps/${b}.json" --data 'data/500_malicious.libsvm' --model_type 'xgboost' --model "models/twitter/${md}.bin" --num_classes 2 --feature_start 0 --out "result/gbdt/adap_${md}_${o}.txt" --adv "adv_examples/gbdt/${md}_${o}.pickle" >! logs/milp_gbdt_adap_${md}_${o}.log &
    ```
  * cost_4:
    ```
    o='obj4';b='bound_obj4'; md='twitter_spam_nature'; python flexible_xgbKantchelianAttack_cost.py -n 100 --weight config/weight/${o}.json -b "config/eps/${b}.json" --data 'data/500_malicious.libsvm' --model_type 'xgboost' --model "models/twitter/${md}.bin" --num_classes 2 --feature_start 0 --out "result/gbdt/adap_${md}_${o}.txt" --adv "adv_examples/gbdt/${md}_${o}.pickle" >! logs/milp_gbdt_adap_${md}_${o}.log &
    ```
