import numpy as np
import os
import subprocess
import sys


def install_smooth_emulator(Smooth_path):
    """
    Function to clone the repository and install the SmoothEmulator software.
    """
    # Clone the repository if not already present
    if not os.path.exists(Smooth_path):
        print("Cloning bandframework repository...")
        subprocess.run("git clone https://github.com/bandframework/bandframework.git", shell=True, check=True)
    else:
        print("SmoothEmulator already exists, skipping cloning.")

    # Copy files
    print("Copying necessary files...")
    subprocess.run(f"cp src/smooth_util/master.cc {Smooth_path}/software/src/smooth/master.cc", shell=True, check=True)
    subprocess.run(f"cp src/smooth_util/master.h {Smooth_path}/software/include/msu_smooth/master.h", shell=True, check=True)
    subprocess.run(f"cp src/smooth_util/pybind_main.cc {Smooth_path}/templates/mylocal/software/pybindstuff/pybind_main.cc", shell=True, check=True)
    subprocess.run(f"cp src/smooth_util/emulator_parameters.txt {Smooth_path}/templates/myproject/smooth_data/smooth_parameters/emulator_parameters.txt", shell=True, check=True)
    
    print("All files copied successfully.")

    # List of directories to run cmake and make in
    directories = [
        "software",
        "templates/mylocal/software/pybindstuff",
        "templates/mylocal/software"
    ]

    # Loop through each directory and run cmake and make
    for directory in directories:
        full_path = os.path.join(Smooth_path, directory)
        print(f"Changing to directory: {full_path}")
        subprocess.run(f"cd {full_path} && cmake . && make", shell=True, check=True)
        print(f"Completed cmake and make in {full_path}")

def fit(x, theta, f, priors, label, order, Lambda=1.0):
    nx = x.shape[0]
    path = os.path.join('smooth_emulators', label)
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    
    smooth_path = os.environ.get('Smooth_PATH')
    subprocess.run(['cp', '-r', f'{smooth_path}templates/myproject/smooth_data', '.'], check=True)
    
    # Write necessary files
    update_emulator(order, Lambda)
    write_observable_info(x, exp_error=np.full(nx, 0.0))
    update_parameters_in_file(priors)
    write_obs_txt(nx, f)
    write_mod_par(theta)
    
    # Tune emulator
    subprocess.run([f'{smooth_path}templates/mylocal/bin/./smoothy_tune'], check=True, capture_output=True, text=True)

    os.chdir('../..')

def update_emulator(order, Lambda):
    filename = 'smooth_data/smooth_parameters/emulator_parameters.txt'
    with open(filename, 'r') as file:
        lines = file.readlines()

    modified_lines = []
    for line in lines:
        if "SmoothEmulator_LAMBDA" in line:
            modified_lines.append(f"SmoothEmulator_LAMBDA {Lambda}\n")
        elif "SmoothEmulator_MAXRANK" in line:
            modified_lines.append(f"SmoothEmulator_MAXRANK {order}\n")
        elif "SmoothEmulator_TrainingFormat training_format_surmise" in line:
            modified_lines.append(line.lstrip())
        elif "SmoothEmulator_TrainingFormat training_format_smooth" in line:
            if not line.startswith("#"):
                modified_lines.append(f"# {line}")  
            else:
                modified_lines.append(line)  
        else:
            modified_lines.append(line)

    # Write the modified lines back to the file
    with open(filename, 'w') as file:
        file.writelines(modified_lines)

def write_observable_info(x, exp_error):
    #Saves observable(the x values) in the first column and their errors in second column
    path = 'smooth_data/Info/'
    filename = 'observable_info.txt'
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, filename), 'w') as out_file:
        for i in range(len(x)):
            out_file.write(f"{x[i]} {exp_error[i]}\n")

def update_parameters_in_file(priors):
    path = 'smooth_data/Info/'
    filename = 'modelpar_info.txt'
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, filename)

    with open(file_path, 'w') as out_file:
        # Write the header line
        out_file.write("# par_name dist_type xmin  xmax\n")
        
        for param_name, prior in priors.items():
            # Access the minimum and maximum from the bilby prior
            xmin = str(prior.minimum)
            xmax = str(prior.maximum)
            out_file.write(f"{param_name} uniform {xmin} {xmax}\n")

def write_mod_par(theta):
    #makes file and saves training parameters. row for each parameter set
    filename = 'TrainingThetas.txt'
    if os.path.exists(filename):
        os.remove(filename)

    with open(filename, 'w') as out_file:
        for row in theta:
            row_str = ' '.join(map(str, row))
            out_file.write(f"{row_str}\n")

def write_obs_txt(nx, f):
    #makes file and saves predictions for training parameters. column for each x value and row for each parameter set.
    filename = 'TrainingObs.txt'
    if os.path.exists(filename):
        os.remove(filename)

    with open(filename, 'w') as out_file:
        for i in range(len(f[0:])):
            for j in range(nx):
                out_file.write(f"{f[i][j]} ")
            out_file.write("\n")

def predict(train_points, label):
    smooth_path = os.environ.get('Smooth_PATH')
    sys.path.insert(0, smooth_path + "templates/mylocal/software/pybindstuff")
    
    work_path = os.environ.get('Work_PATH')
    path = f"/{work_path}/smooth_emulators/{label}/"
    os.chdir(path)
    import emulator_smooth
    smoothmaster = emulator_smooth.emulator_smooth() 
    
    smooth_train = []
    smooth_train_error = []

    NObs = smoothmaster.GetNObs() 

    for param in train_points:
        X = param
        Y = np.zeros(NObs,dtype='float')
        SigmaY = np.zeros(NObs,dtype='float')
        theta = smoothmaster.GetThetaFromX(X)
        smoothmaster.TuneAllY()
        for iY in range(NObs):
            Y[iY], SigmaY[iY] = smoothmaster.GetYSigmaPython(iY, theta)
        smooth_train.append(Y)
        smooth_train_error.append(SigmaY)

    
    return np.array(smooth_train), np.array(smooth_train_error)