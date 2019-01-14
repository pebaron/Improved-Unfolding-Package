# Unfolding

1.) Install ROOT
2.) Install Fully Bayesian Unfolding package - FBU available here https://github.com/gerbaudo/fbu
3.) Install RooUnfold package - http://hepunx.rl.ac.uk/~adye/software/unfold/RooUnfold.html

In case you would like to use FBU Regularization, download old FBU package https://github.com/gerbaudo/fbu/releases/tag/v0.0.2
and use older version of ROOT e.g.16.04 and python 2.7 and pymc (not pymc3)

1.) first just test the script writing:

python Unfolding.py

In input.root are saved 4 histograms = data, backgournd, particle level and migration matrix

2.) TO RUN THE SCRIPT you need 4 inputs data, backgournd, particle level and migration matrix, Migration matrix has detector level on X axis, and particle level on Y axis

example:

python Unfolding.py --rfile_data "/home/petr/GitLab/unfolding/input.root" --rfile_particle "/home/petr/GitLab/unfolding/input.root" --rfile_matrix "/home/petr/GitLab/unfolding/input.root" --rfile_background "/home/petr/GitLab/unfolding/input.root" --h_data "h_data" --h_particle "h_particle" --h_matrix "h_matrix" --h_background "h_background"
 
 3.) You can use option to rebin if you wish:

 example:

 python Unfolding.py --nrebin 2

4.) If your posteriors are drown red, increase the number of iterations by:

python Unfolding.py --iterations 2

or 

python Unfolding.py --iterations 3

until you get satisfied results.

5.) Script generates png and root files
