# Affinis

A Recurrent Neural Network SubDomain Discovery Tool

Affinis aims to find potentially hidden or forgotten subdomains through the use of Natural Language Processing and the Keras LSTM RNN API.
It will read in a file of already discovered subdomains for the target and generate its own list of potential subdomains the target may have.
It then verifies the existence of those subdomains with a quick DNS lookup and will ultimately tell you if it has made any discoveries.

![running Affinis](https://github.com/Jetp1ane/Affinis/raw/master/images/run.PNG)

![output Affinis](https://github.com/Jetp1ane/Affinis/raw/master/images/generated.png)

I am by no means an expert on neural networks, but this project is my attempt to learn. Expect it to grow and mold into something more sophisiticated.
You can read some more specifics on the project on my blog: [Phoenix-sec.io](https://phoenix-sec.io)

# To Run:

Requirements Installation:
  - **pip install -r requirements.txt**

Command:
  - **python3 Affinis.py [domain] [number to generate] [path to existing subdomains file]**
  - **python3 Affinis.py google.com 500 /tmp/google_subs.txt**
  
It can certainly take awhile to run if you run tensorflow off your CPU, but you can expedite things by configuring Keras for GPU: [GPU Integration](https://wandb.ai/authors/ayusht/reports/Using-GPUs-With-Keras-A-Tutorial-With-Code--VmlldzoxNjEyNjE)
