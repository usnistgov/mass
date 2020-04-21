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

What is in a record, lets see

.. testcode::

  print(off[0])
  print(off.dtype)

.. testoutput::

  (1000, 496, 556055239, 1544035816785813000, 10914.036, 22.124851, 10929.741, -10357.827, 10609.358, [-47.434967,  -8.839941])
  [('recordSamples', '<i4'), ('recordPreSamples', '<i4'), ('framecount', '<i8'), ('unixnano', '<i8'), ('pretriggerMean', '<f4'), ('residualStdDev', '<f4'), ('pulseMean', '<f4'), ('derivativeLike', '<f4'), ('filtValue', '<f4'), ('extraCoefs', '<f4', (2,))]

recrods of off files numpy arrays with dtypes, which contain may filed. The exact set of fields depends on the off file version as they are still under heavy development. The projector coefficients are stored in "pulseMean", "derivativeLike", "filtValue" and "extraCoefs".

Now lets open the same file as a ``Channel``.

.. testcode::

  from mass.off import ChannelGroup, getOffFileListFromOneFile, Channel, labelPeak, labelPeaks
  data = ChannelGroup(getOffFileListFromOneFile("../mass/off/data_for_test/20181205_BCDEFGHI/20181205_BCDEFGHI_chan1.off", maxChans=2))
  ds = data[1]
  ds.plotHist(np.arange(0,40000,10),"filtValue")
  plt.savefig("img/hist1.svg")

Here we see two images next to each other

|pic1| |pic2|

.. |pic1| image:: img/hist1.svg
  :width: 45%

.. |pic2| image:: img/hist1.svg
  :width: 45%

And another way to do two images side by side

.. image:: img/hist1.svg
  :width: 45%

.. image:: img/hist1.svg
  :width: 45%