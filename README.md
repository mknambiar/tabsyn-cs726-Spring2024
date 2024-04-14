# tabsyn-cs726-Spring2024
A modified version of the tabsyn project presented in ICLR for course CS726
This is the original project - <a href="https://github.com/amazon-science/tabsyn"> </a>

Refer here - most instructions are the sam

In this README we will just tell you how to install and run the project.

First of all you need to install conda. Remember that you do not need to be a sudo user.
This URL tells you how to install and use conda without root access
<a href="https://csmasterme.wordpress.com/2021/01/22/installing-anaconda-locally-for-a-user-in-remote-server-without-sudo-permission/"> </a>


Just replace these instructions
"Create another environment for the quality metric (package "synthcity")

conda create -n tabsyn python=3.10
conda activate tabsyn

pip install synthcity
pip install category_encoders"

with these
"Create another environment for the quality metric (package "synthcity")

conda create -n metrics python=3.10
conda activate metrics

pip install synthcity
pip install category_encoders"

Plus install any module python tells you is missing using pip install.

Ensure that you are in the metrics environment before running eval_quality.py.
This can be done using "conda activate metrics"

For everything else you can do it in tabsyn conda environment
