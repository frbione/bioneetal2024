# Estimating total organic carbon of potential source rocks in the Espírito Santo Basin, SE Brazil, using XGBoost

Author: [frbione](https://github.com/frbione)

Bione et al. (2024) scientific paper repository.

Reference: 
BIONE F.R.A. et al. 
Estimating total organic carbon of potential source rocks in the Espírito Santo Basin, SE Brazil, using XGBoost. 
Marine and Petroleum Geology, v. 162, 106765, 2024. 
https://doi.org/10.1016/j.marpetgeo.2024.106765


___
# ⚙️Configuring the environment

### Creating the venv and installing the dependencies
In terminal, run: 

`python -m venv venv` →	  
`cd venv\Scripts` →   
`activate.bat` →   
`cd ..\..` →   
`pip install -r requirements.txt`  

___
# 📖 Create the train-test dataframe 
To create the train-test dataframe from the originally-compiled full dataframe: 
* Open the [Data_preparation](https://github.com/frbione/bioneetal2024/blob/main/notebooks/Data_preparation.ipynb) 
notebook, and run all four steps to generate the `train-test dataframe`

___
# 📈 Reproducing and visualizing the results
To reproduce the results using the already tuned model parameters and visualize the results:
* Go to the [Reproduce_and_generate_figs](https://github.com/frbione/bioneetal2024/blob/main/notebooks/Reproduce_and_generate_figs.ipynb) notebook, and follow the instructions provided in the notebook.

___
___

# ▶️ Tune your own models

If you want to run your own models using this approach, bear in mind you must provide a compatible dataframe.
Thus, it is very likely that some code adaptations will be required, such as renaming features/targets, data imputation 
parameters or any other feature engineering technique you wish to include.


### Installing pySpark (Windows)

<span style="color: #E7816B;">Optional. Do this in case you want to run parameter tuning for your own models.</span>

* Download JDK from [this link](https://www.oracle.com/in/java/technologies/downloads/#jdk19-windows), and install it;
* Download Spark from [this link](https://spark.apache.org/downloads.html), then extract the tar file to a directory (e.g., <span style="background-color: #2b7b80">C:\spark</span>);
* Download hadoop from [this link](https://github.com/steveloughran/winutils/blob/master/hadoop-2.7.1/bin/winutils.exe), then add the `winutils` file to a directory (e.g., <span style="background-color: #2b7b80">C:\hadoop\bin</span>);


* Configure the `Environment Variables` by adding the following:
  *    `JAVA_HOME` - (e.g., <span style="background-color: #2b7b80">C:\java\jdk</span>)
  *    `HADOOP_HOME` - (e.g., <span style="background-color: #2b7b80">C:\hadoop</span>)
  *    `SPARK_HOME` - (e.g., <span style="background-color: #2b7b80">C:\spark\spark-3.3.2-bin-hadoop2</span>)
  *    `PYSPARK_HOME` - (e.g., <span style="background-color: #2b7b80">..\venv\lib\site-packages\pyspark</span>)


* Finally, add the following to Path:
  * `%JAVA_HOME%\bin`
  * `%HADOOP_HOME%\bin`
  * `%SPARK_HOME%\bin`

After installing pySpark, you can run the [model_run.py](https://github.com/frbione/bioneetal2024/tree/main/modules/model_run.py) script, passing your own dataframe.