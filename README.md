Odysseus
========

Odysseus is a framework for designing and running simulations of fluid flows, relying on the Dedalus code.
It can be seen as a wrapper wround Dedalus which does not aim to be as general but to provide a simple way to run simulations of classical fluid dynamics equations without having to construct everything from scratch using Dedalus.
As a more specialized framework, it comes with analysis tools specific to the problems at hand.

I am not sure that it will be more than a launcher that I use for my personal simulations, for convenience and to ensure reproducibility, but I'd be happy if it could be useful to other researchers.

Install
-------

First [install Dedalus](https://dedalus-project.readthedocs.io/en/latest/pages/installation.html) on your machine. Then type
```
conda activate dedalus
git clone git@github.com:cbherbert/odysseus.git
cd odysseus
pip install -e .
```
