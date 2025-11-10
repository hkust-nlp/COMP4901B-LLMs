# COMP4901B – School Cluster Usage Guide

This guide provides a step-by-step walkthrough for utilizing the department's computing cluster for your coursework.


### By default, School Cluster use C shell (csh) not Bash. However, all the setup commands we provided are written for Bash, so before running any setup commands, please switch to Bash by typing `bash`. 


## 1. Accessing the Cluster

The department's cluster consists of a head node and multiple compute nodes. Each compute node has one 2080 Ti GPU. The head node serves as the entry point, while the compute nodes are where you will run your computational tasks. **It's important to remember that the head node's disk is separate from the compute nodes' disks, so you should only use the head node for logging in and managing your jobs, and perform your actual work on the compute nodes.** 

**Rules**
- This cluster is dedicated for this course, and it will continue supporting our other assignments as well
- Each student is permitted to log on to only one compute node at a time, this is constrained by the cluster mechanism.
- **Please act as a good citizen and log out of the compute node when you are not using it.** Because others cannot log onto the same compute node when you are using it.
- The number of compute nodes is smaller than the total number of students, but should be enough if you all are not working at the same time. This also means the cluster may be crowded near the due date, so please start early.

**Step 0: Activate CSD Accounts**

Every student on the CSE UG course has a CSE UG UNIX account, which is required to use this cluster. Please open https://cssystem.cse.ust.hk/UGuides/activation.html to activate your account. Remember to check the box for "Unix account at UG domain".

**Step 1: Logging into the Head Node**

To begin, you will need to log in to the head node using your CSE UG UNIX account password. Open a terminal and use the following command:

```bash
ssh your_username@ugcnode01.cse.ust.hk
```

Replace `your_username` with your actual CSE username, and you need to input the password of your CSE UG UNIX account.

**Step 2: Checking Your Usage and Finding an Available Compute Node**

Once logged in to the head node, run the following command to check if you have an active session and to find an available compute node:

```bash
check_usage
```

This command will inform you of any active sessions. If you don't have one, it will list the available `ughostXX` nodes. Each student is permitted to log on to one `ughostXX` at a time.

**Step 3: Logging into a Compute Node**

From the head node, you can now log in to an available compute node. For instance, if `ughost01` is available, you would use the following command:

```bash
ssh ughost01
```

## 2. Setting Up Your Workspace and Project

All your data and project files should be stored in the `/store/comp4901b/` directory on the compute node.

**Step 1: Creating Your Personal Directory**

#### It is crucial to create a subdirectory with your username to keep your files organized and separate from other students' work.

```bash
mkdir /store/comp4901b/your_username

# change permission so that the directory is only accessible by you
chmod 700 /store/comp4901b/your_username
```

**Step 2: Cloning the Homework Project**

Navigate into your newly created directory and clone the homework project from its repository.

```bash
cd /store/comp4901b/your_username
git clone <repository_url>
```

Replace `<repository_url>` with the actual URL of the homework project's repository.


**Step 3: Configuring the CUDA Environment on the Compute Node**

To use the GPUs on the compute nodes, you need to configure your shell environment to locate the CUDA toolkit. This is done by adding environment variables to your `.bashrc` file, which ensures they are set every time you log in.

On the compute node (e.g., ughost01), run the following command to append the necessary configuration to your `~/.bashrc` file:

```bash
cat <<EOF >> ~/.bashrc

# CUDA Environment Setup
export CUDA_HOME="/usr/local/cuda-12.6"
export PATH="/usr/local/cuda-12.6/bin:\$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64:\$LD_LIBRARY_PATH"
EOF
```
After running the command, you must apply the changes by either logging out and back in, or by running `source ~/.bashrc`.

To confirm that CUDA is correctly set up, run the command `nvcc -V`.  You should see output indicating **Cuda compilation tools, release 12.6**.


## 3. Environment Setup with Miniconda

To ensure you have the correct software and library versions for your assignments, you will use Miniconda to create an isolated environment.

**Step-by-Step Command Explanation:**

1.  **Download the Miniconda Installer** This command downloads the latest Miniconda installer script for a 64-bit Linux system.
    ```bash
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    ```

2.  **Run the Installer:** This command executes the installer script.
    ```bash
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /store/comp4901b/your_username/miniconda3
    ```

3.  **Source the Conda Profile:** This command updates your shell's environment variables so it can find the `conda` command. Again, **replace `your_username` with your username.**
    ```bash
    source /store/comp4901b/your_username/miniconda3/etc/profile.d/conda.sh
    ```

4.  **Initialize Conda for Bash:** This command modifies your shell's startup script to automatically initialize Conda whenever you start a new terminal session.
    ```bash
    conda init bash
    ```
    After running this, you may need to close and reopen your terminal for the changes to take effect.

5.  **Create a Conda Environment:** This command creates a new, isolated environment named `comp4901b-hw2` with Python version 3.10.
    ```bash
    conda create -n comp4901b-hw2 python=3.10
    ```

6.  **Activate the Environment:** This command activates the newly created environment. Any packages you install will be specific to this environment.
    ```bash
    conda activate comp4901b-hw2
    ```

7.  **Run the Setup Script:** This command executes a setup script provided with the homework, which will likely install all the necessary Python packages for your assignment.
    ```bash
    bash setup.sh
    ```

You are now ready to begin your work on the school's cluster. Remember to always work within your designated directory on the compute nodes and to use your Conda environment for all tasks.

## 4. Setting Up Passwordless SSH Access and VScode (Optional but recommended)
Directly editing code files on the compute node needs to use `vim` that is not very friendly. Generally you can:

1. Work on your local machine and copy the code files to the compute node.
2. Git fork the repo and uses git push and pull to sync the code files between your local machine and the compute node.
3. Connect to the compute node using vscode or cursor to directly code the remote files in your local machine. This is the most convenient way but needs some setup, which we detail below. 

To streamline your workflow, you can set up SSH keys to log in to the cluster without a password. This involves generating a key pair on your local computer and copying the public key to the cluster's head and compute nodes.

Because the head node does not have port forwarding enabled, we will use `netcat` (`nc`) as a proxy to reach the compute nodes from your local machine.

**Step 1: Generate an SSH Key on Your PC**

If you don't already have an SSH key, open a terminal on your local computer and run the following command.

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

Replace `"your_email@example.com"` with your actual email address. You can press Enter to accept the default file location and leave the passphrase empty for passwordless login.

**Step 2: Copy Your Public Key to the Head and Compute Nodes**

Next, you need to copy the public key you just generated to both the head node and the compute node you will be using. You will be prompted for your password for each of these commands.

*   **Copy key to the head node (`ugcnode01`):**

    ```bash
    ssh-copy-id your_username@ugcnode01.cse.ust.hk
    ```

*   **Copy key to a compute node (e.g., `ughost01`):**

    This command is more complex as it uses the head node as a proxy. Make sure to replace `your_username` with your username and `ughost01` with your assigned compute node if it's different.

    ```bash
    ssh-copy-id -o ProxyCommand='ssh -q ugcnode01.cse.ust.hk nc %h %p' your_username@ughost01.cse.ust.hk
    ```

**Step 3: Configure Your Local SSH Client**

To make the connection seamless, add the following configuration to the `config` file located in your `~/.ssh/` directory on your local PC. If the file doesn't exist, you can create it.

```
# ~/.ssh/config

# Alias for the Head Node
Host C
  HostName ugcnode01.cse.ust.hk
  User your_username
  ForwardAgent yes

# Alias for the Compute Node, accessed via the Head Node
Host uggpu
  HostName ughost01.cse.ust.hk
  User your_username
  ProxyCommand ssh -q C nc %h %p
```

**Important:** Remember to replace `your_username` with your CSE username in both host configurations. **You may need to change ughost01 to another compute node depending on the availability.**

**Step 4: Connect Directly Without a Password**

With this configuration in place, you can now connect directly to the compute node from your local machine's terminal or configure tools like VSCode's Remote-SSH extension to use the `uggpu` host.

```bash
ssh uggpu
```

<<<<<<< HEAD
This command will now connect you directly to `ughost01` through `ugcnode01` without prompting for a password. This way, you can directory connect with uggpu using vscode or cursor.
=======
This command will now connect you directly to `ughost01` through `ugcnode01` without prompting for a password. This way, you can directory connect with uggpu using vscode or cursor.
>>>>>>> 846ec3382b0c09ac1adc60e65c032bec5399a431
