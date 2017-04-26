## IPython Demonstrations

In the directory `mass/src/mass/demo` there are a small number of python scripts
written especially to serve as "IPython Demos" and run in that framework. The
basic usage is like this, assuming you're in an IPython session:

```python
import mass.demo
massdemo = mass.demo.demos["intro.py"]
massdemo()  # then read terminal and hit enter
massdemo()  # then read terminal and hit enter
massdemo()  # then read terminal and hit enter
# Repeat several times until the demo tells you it's done.
```

The demos that exist at the time of this writing include:

* `intro.py`: A basic introduction to most of the main features of a simple data
analysis in MASS.
* `fitting_demo.py`: A demonstration of fitting Gaussian, Voigt, and
multiple-Lorentzian functions to histogram data.
* `fitting_fluorescence.py`: A demonstration of fitting the Mn K-alpha complex
to histogram data.
* `cuts.py`: Walks you through a few of the complex features involving decisions
about good/bad pulses ("boolean cuts") and how we label pulses by category
("categorical cuts"). See also [Cuts.md](Cuts.md) in this directory.
* `full_analysis_example.py`: An example like `intro.py`, except with more pulses
and including a Mn K-alpha fit to estimate energy resolution. Requires installing
a git repository `ReferenceMicrocalFiles.jl` to provide the sample data.
