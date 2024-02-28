
### Data
Included data were obtained from open data repositories. Every set corresponds to different machine learning task. Names and sources are described in the table below.

 | Data                  | Data source                                                             | Task |
|-----------------------|-------------------------------------------------------------------------| ---- | 
| [bank_marketing](https://github.com/malwina0/cattleia/tree/AutoMLConf24/examples/bank_marketing)        | [UCI Machine Learning Repository - Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)|binary classification |
| [artificial_characters](https://github.com/malwina0/cattleia/tree/AutoMLConf24/examples/artificial_characters) | [OpenML - Dataset 1459](https://www.openml.org/search?type=data&status=active&id=1459)| multiclass classification |
| [life_expectancy](https://github.com/malwina0/cattleia/tree/AutoMLConf24/examples/life_expectancy)       | [Kaggle - Life Expectancy](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who) |  regression|

Every model is split into train and test set. Preprocessing that was needed for frameworks to train properly is already applied to data. 

### Models

Models included in each directory were trained using three AutoML packages namely: auto-sklearn, AutoGluon and FLAML. For auto-sklearn and FLAML files in pickle format with corresponding names are included. For AutoGluon, the zip archive is presented. All available models were trained on a train subsets of data. 
