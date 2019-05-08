The code for this assigment has been taken from https://github.com/XuetingChen/cs7641-assignment-4
The Readme has also been copied from the same source.

My modified code is uploaded at https://drive.google.com/open?id=16Rioue3mSCKkk1rIIUfFx1o4SHgKKtrV


To run, import the project into Eclipse as a maven project. Under the ./src/burlap/assignment4/ directory, find files named EasyGridWorldLauncher.java or HardGridWorldLauncher_6X6. Each of these are the main entry point to run a different grid problem:

* EasyGridWorldLauncher.java: Runs an analysis of a simple 5x5 grid problem with one terminal state and obstructions
* HardGridWorldLauncher_6x6.java: Runs an analysis of a simple 16x16 grid problem with one terminal state and obstructions.

##To compile: Right click on the root folder of the project, click on update project. 

##To run:
1. While inside the directory structure ./src/burlap/assignment4/ right-click on the appropriate worldLauncher.java.
2. Go to the “Run As…” section and select “Java Application.
3. All three algorithms will run and the aggregate analysis and optimal policies will be printed to the console.

##Sample Output
This is the sort of output you get out of the box by running the HardGridWorldLauncher as a Java Application:

```
/////Hard Grid World Analysis/////

This is your grid world:
[0,0,0,0,0,0,0,0,0,0,0]
[0,0,0,0,0,0,0,0,0,0,0]
[0,0,0,0,0,0,0,0,0,0,0]
[0,0,0,1,1,1,1,1,0,0,0]
[0,0,0,1,0,0,0,1,0,0,0]
[0,0,0,1,0,0,0,1,0,0,0]
[0,0,0,1,0,0,0,1,0,0,0]
[0,0,0,1,1,1,1,1,0,0,0]
[0,0,0,0,0,0,0,0,0,0,0]
[0,0,0,0,0,0,0,0,0,0,0]
[0,0,0,0,0,0,0,0,0,0,1]



//Value Iteration Analysis//
Passes: 1
...
Passes: 20
Value Iteration,3752,2228,1150,49,33,31,29,36,31,28,39,29,25,30,28,26,23,23,44,31

This is your optimal policy:
num of rows in policy is 11
[>,>,>,>,>,>,>,>,>,>,>]
[>,>,>,>,>,>,>,>,>,>,^]
[>,>,>,>,>,>,>,>,^,^,^]
[^,^,^,*,*,*,*,*,^,^,^]
[^,^,^,*,*,*,*,*,^,^,^]
[^,^,^,*,*,*,*,*,^,^,^]
[^,^,^,*,*,*,*,*,^,^,^]
[^,^,^,*,*,*,*,*,^,^,^]
[^,^,^,>,>,>,>,>,^,^,^]
[^,^,>,>,>,>,>,>,^,^,^]
[^,>,>,>,>,>,>,>,^,^,*]



Num generated: 1500; num unique: 95
//Policy Iteration Analysis//
Total policy iterations: 1
...
Total policy iterations: 20
Policy Iteration,11375,8124,1514,626,2549,528,214,46,23,25,30,25,21,28,25,23,26,25,25,24
Passes: 19

This is your optimal policy:
num of rows in policy is 11
[>,>,>,>,>,>,>,>,>,>,v]
[>,>,>,>,>,>,>,>,>,^,^]
[>,>,>,>,>,>,>,>,^,^,^]
[^,^,^,*,*,*,*,*,^,^,^]
[^,^,^,*,*,*,*,*,^,^,^]
[^,^,^,*,*,*,*,*,^,^,^]
[^,^,^,*,*,*,*,*,^,^,^]
[^,^,^,*,*,*,*,*,^,^,^]
[^,^,^,>,>,>,>,>,^,^,^]
[^,^,>,>,>,>,>,>,^,^,^]
[^,^,>,>,>,>,>,>,^,^,*]



//Q Learning Analysis//
Q Learning,301,571,191,194,150,302,129,64,137,364,113,114,69,141,155,58,108,89,71,78
Passes: 19

This is your optimal policy:
num of rows in policy is 11
[>,>,^,>,>,>,v,>,^,>,>]
[^,^,>,>,v,>,>,^,>,^,^]
[>,<,>,>,>,^,>,>,>,^,v]
[^,^,^,*,*,*,*,*,^,v,^]
[<,^,<,*,*,*,*,*,v,^,v]
[^,>,<,*,*,*,*,*,>,^,^]
[<,v,^,*,*,*,*,*,^,<,^]
[<,>,<,*,*,*,*,*,>,v,^]
[^,>,>,>,<,>,>,>,^,>,>]
[>,>,^,^,>,v,^,^,>,^,^]
[^,^,v,v,>,>,<,v,<,v,*]



//Aggregate Analysis//

The data below shows the number of steps/actions the agent required to reach 
the terminal state given the number of iterations the algorithm was run.
Iterations,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20
Value Iteration,3752,2228,1150,49,33,31,29,36,31,28,39,29,25,30,28,26,23,23,44,31
Policy Iteration,11375,8124,1514,626,2549,528,214,46,23,25,30,25,21,28,25,23,26,25,25,24
Q Learning,301,571,191,194,150,302,129,64,137,364,113,114,69,141,155,58,108,89,71,78

The data below shows the number of milliseconds the algorithm required to generate 
the optimal policy given the number of iterations the algorithm was run.
Iterations,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20
Value Iteration,224,36,39,44,32,53,67,104,81,86,88,135,121,208,149,100,86,115,102,106
Policy Iteration,18,32,20,51,26,32,39,50,65,51,97,107,111,90,64,80,71,82,162,164
Q Learning,37,50,34,76,29,73,52,44,65,60,28,45,32,23,53,36,34,29,40,40

The data below shows the total reward gained for 
the optimal policy given the number of iterations the algorithm was run.
Iterations,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20
Value Iteration Rewards,-3650.0,-2126.0,-1048.0,53.0,69.0,71.0,73.0,66.0,71.0,74.0,63.0,73.0,77.0,72.0,74.0,76.0,79.0,79.0,58.0,71.0
Policy Iteration Rewards,-11273.0,-8022.0,-1412.0,-524.0,-2447.0,-426.0,-112.0,56.0,79.0,77.0,72.0,77.0,81.0,74.0,77.0,79.0,76.0,77.0,77.0,78.0
Q Learning Rewards,-199.0,-469.0,-89.0,-92.0,-48.0,-200.0,-27.0,38.0,-35.0,-262.0,-11.0,-12.0,33.0,-39.0,-53.0,44.0,-6.0,13.0,31.0,24.0

```

##To plot: 
1.) Plotting policy maps: Copy above output from "The data below shows.." and paste it "small" or "large" file found in ./src/burlap/assignment4/plottingLib/ folder based on the type of problem. Trigger the appropriate experiment in the constructor of the plotting class. The names of the experiment sub-routines are self-explanatory.
2.) For plotting Q-Learning Experiments: For instance, if you want to plot varying epislon experiment, run the appropriate functions in XXXXGridWorldLauncher.java (All of them are commented out, you can uncomment the ones you want to run). Then copy of the rewards array for Q-Learning from the output and paste it in 'epsilonRewards' file and the time array in 'epsilonTime' file. Then run the necessary subroutines in plot.py. The same workflow applies for varying learning rate and discount reward factor.

