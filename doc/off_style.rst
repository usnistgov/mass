OFF Style Mass Analysis
=======================

This page is primarily intended to test out the features of Sphinx, including doctest, and the use of images generated from those tests in the documentation. It may or may not be expanded in the future depending on how much value it seems to add vs how much of a pain it is. The workflow is not great... you run ``make doctest;make html;open _build/html/index.html`` or just ``make doctest`` after each set of edits and see if things worked.

Exploring an OFF File
---------------------
Here we open an OFF file and plot a reconstructed record.

.. testcode::

  import mass
  import numpy as np 
  import pylab as plt

  off = mass.off.OffFile("../mass/off/data_for_test/20181205_BCDEFGHI/20181205_BCDEFGHI_chan1.off")
  x,y = off.recordXY(0)
  
  plt.figure()
  plt.plot(x,y)
  plt.xlabel("time (s)")
  plt.ylabel("reconstructed signal (arbs)")
  plt.savefig("img/offxy.svg")

.. image:: img/offxy.svg
  :width: 600

What is in a record?

.. testcode::

  print(off[0])
  print(off.dtype)

.. testoutput::

  (1000, 496, 556055239, 1544035816785813000, 10914.036, 22.124851, 10929.741, -10357.827, 10609.358, [-47.434967,  -8.839941])
  [('recordSamples', '<i4'), ('recordPreSamples', '<i4'), ('framecount', '<i8'), ('unixnano', '<i8'), ('pretriggerMean', '<f4'), ('residualStdDev', '<f4'), ('pulseMean', '<f4'), ('derivativeLike', '<f4'), ('filtValue', '<f4'), ('extraCoefs', '<f4', (2,))]

recrods of off files numpy arrays with dtypes, which contain may filed. The exact set of fields depends on the off file version as they are still under heavy development. The projector coefficients are stored in "pulseMean", "derivativeLike", "filtValue" and "extraCoefs". You can access 


Basic Analysis with a ``ChannelGroup``
--------------------------------------

.. testcode::

  from mass.off import ChannelGroup, getOffFileListFromOneFile, Channel, labelPeak, labelPeaks
  data = ChannelGroup(getOffFileListFromOneFile("../mass/off/data_for_test/20181205_BCDEFGHI/20181205_BCDEFGHI_chan1.off", maxChans=2))
  data.learnResidualStdDevCut(plot=True)
  plt.savefig("img/residualStdDevCut.svg")
  ds = data[1]
  ds.plotHist(np.arange(0,40000,10),"filtValue")
  plt.savefig("img/hist1.svg")

.. image:: img/residualStdDevCut.svg
  :width: 45%

.. image:: img/hist1.svg
  :width: 45%

Here we opened all the channels that have the same base filename, plus we opened the ``_experiment_state.txt`` that defines states. States provide a convenient way to seperate your data into different chunks by time, and a generally assigned during data aquistion, but you can always make a new ``_experiment_state.txt`` file to split things up a different way.

.. testcode::

  from mass.calibration import _highly_charged_ion_lines
  data.setDefaultBinsize(0.5)
  ds.calibrationPlanInit("filtValue")
  ds.calibrationPlanAddPoint(2128, "O He-Like 1s2s+1s2p", states="CO2")
  ds.calibrationPlanAddPoint(2421, "O H-Like 2p", states="CO2")
  ds.calibrationPlanAddPoint(2864, "O H-Like 3p", states="CO2")
  ds.calibrationPlanAddPoint(3404, "Ne He-Like 1s2s+1s2p", states="Ne")
  ds.calibrationPlanAddPoint(3768, "Ne H-Like 2p", states="Ne")
  ds.calibrationPlanAddPoint(5716, "W Ni-2", states=["W 1", "W 2"])
  ds.calibrationPlanAddPoint(6413, "W Ni-4", states=["W 1", "W 2"])
  ds.calibrationPlanAddPoint(7641, "W Ni-7", states=["W 1", "W 2"])
  ds.calibrationPlanAddPoint(10256, "W Ni-17", states=["W 1", "W 2"])
  # ds.calibrationPlanAddPoint(10700, "W Ni-20", states=["W 1", "W 2"])
  ds.calibrationPlanAddPoint(11125, "Ar He-Like 1s2s+1s2p", states="Ar")
  ds.calibrationPlanAddPoint(11728, "Ar H-Like 2p", states="Ar")
  # at this point energyRough should work
  ds.plotHist(np.arange(0, 4000, 1), "energyRough", coAddStates=False)
  plt.savefig("img/hist2.svg")

.. image:: img/hist2.svg
  :width: 45%