The src uses pandas, scikit and numpy to run. So please set up the development environment by running the following commands.
1.) virtualenv venv (If you don't have virtualenv you have to install it using apt-get or brew based on your OS)
2.) source venv/bin/activate
3.) pip install -r requirements.txt

Note: All the above commands have to be run in the directory in which this README.txt is present.

Downloading the source code and the data
=> Please this link to download the code and the data. Then follow the instructions above to set up the developmente environment.


Now you can navigate to src folder and run the supervisedLearning.py file. You should be able to see the graphs in the corresponding folders in the src directory.


To change dataset
=> change the dataset parameter in line 786

To control which classifers to run
=> Select the appropriate classifers in the runExperiment sub routine

