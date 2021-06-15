* [PHT_Synth_Data_Gen](https://github.com/CaspervanAarle/PHT_Synth_Data_Gen)
* [PHT_Server](https://github.com/CaspervanAarle/PHT_Server) (You are here)
* [PHT_Node](https://github.com/CaspervanAarle/PHT_Node)

# PHT_Server
This PHT_Server implements Federated Stochastic Gradient Descent as an aggregation method to apply Linear Regression. Aggregators and Classifiers can be altered easily. Due to the need for simulating numerous Personal Data Stores (PDS), importing big libraries is omitted. A semi-privacy-preserving Homomorphic Encrypted Standardization method and AdaGrad is included for better convergence.



## Usage
### Server
Open a terminal: ```cd src``` ```python main.py```
By starting ```python main.py``` without any arguments, you can choose between existing 'config' files in the settings directory. The interface also provides the possibility to create a new config file. A config file can also be manually created. This file contains info on locations of multiple PDS's. 
```
{
  "lockers": [{"locker_ip": "192.168.0.24", "host_port": "5050"}, {"locker_ip": "192.168.0.24", "host_port": "5051"}], 
  "config_name": "experiment"
}
```
A second 'learnconfig' file must be created (to be implemented) or chosen, which includes all hyperparameters for the learning session. Manual creation is also possible.
```
{
	"config_name": "experiment",
	"learning_rate": 0.05,
	"max_iter": 200,
	"var_list": ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"],
	"target_list":	["RMSD"],
	"regressor": "LinReg", #LinReg/LogReg
	"mode": "NORMAL", #NORMAL/SHUFFLESPLIT
	"optimizer": "AdaGrad", #AdaGrad/SGD
	"standardization": true,
	"calc_train_loss": true,
	"calc_test_loss": true
}
```

### Local Experiment 
The  ```experiment.py``` generates a config file containing locations of an amount of local PDS's. Afterward, it initializes the server with this config file and a learnconfig file containing hyperparameters for the learning session. Examples can be found in the settings directory.
