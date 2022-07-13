# Affinis

A Recurrent Nueral Network SubDomain Discovery Tool

Affinis aims to find potentially hidden or forgotten subdomains through the use of Natural Language Processing and the Keras LSTM RNN API.
It will read in a file of already discovered subdomains for the target and generate its own list of potential subdomains the target may have.
It then verifies the existence of those subdomains with a quick DNS lookup and will ultimately tell you if it has made any discoveries.

I am by no means an experrt on neural networks, but this project is my attempt to learn. Expect it to grow and mold into something more sophisiticated.
You can read some more specifics on the project on my blog: [Phoenix-sec.io](https://phoenix-sec.io)

# To Run:
**Command:**
  - python3 Affinis.py <domain> <number to generate> <path to existing subdomains file> 
