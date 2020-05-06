OFF Style Mass Analysis
=======================

.. warning:: Off style analysis is in development, things will change.


Motivation
----------
OFF files and the associated analysis code and methods were developed to address the following goals:

- High quality spectra available immediatley to provide feedback during experiments
- Faster analysis, for ease of exploratory analysis and to avoid data backlogs
- Improve on the mass design (eg avoid HDF5 file corruption, easier exploratory analysis, simpler code)
- Don't always start from zero, we generally run the same array on the same spectromter
- Handle experiments with more than one sample (eg calibration sample followed by science sample)

In the previous "mass-style" analysis, as described in Joe Fowler's "The Practice of Pulse Processing" paper, 
complete data records are written to disk (LJH files), then analysis starts with generatting optimal filters. 
In mass-style analysis, new filters are generated for every dataset, and varying experimental conditions (eg sample) 
are diffuclt to handle, and often handled by writing different files for each condition. 

In off-style analysis we start with filters generated exactly the same way, then add more filters that can represent
how the pulse shape changes with energy. Data is filtered as it is taken, and only the coefficients of these filter
projections are written to disk in OFF files. For now, we typically write both OFF and LJH files just to be safe.

Making Projectors and ljh2off
-----------------------------
The script ``make_projectors`` will make projectors and write them to disk in a format ``dastardcommander`` and ``ljh2off`` can use. 
The script ``ljh2off`` can generate off files from ljh files, so you can use this style of analysis on any data, or change your projectors. 
Call either with a ``-h`` flag for help, also all the functionality is available through functions in ``mass``.

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
  plt.plot(off.basis[:,::-1]) # put the higher signal to noise components in the front
  plt.xlabel("sample number")
  plt.ylabel("weight")
  plt.legend(["pulseMean","derivativeLike","pulseLike","extra0","extra1"])
  plt.title("basis components")

  plt.figure()
  plt.plot(x,y)
  plt.xlabel("time (s)")
  plt.ylabel("reconstructed signal (arbs)")
  plt.title("example reconstructed pulse")

.. testcode::
  :hide:

  plt.savefig("img/basis.png");plt.close()
  plt.savefig("img/offxy.png");plt.close()

.. image:: img/basis.png
  :width: 45%

.. image:: img/offxy.png
  :width: 45%

.. warning:: The basis shown here was generated with obsolete algorithms, and doesn't look as good as a newer basis will. I should replace it.

What is in a record?

.. testcode::

  print(off[0])
  print(off.dtype)

.. testoutput::

  (1000, 496, 556055239, 1544035816785813000, 10914.036, 22.124851, 10929.741, -10357.827, 10609.358, [-47.434967,  -8.839941])
  [('recordSamples', '<i4'), ('recordPreSamples', '<i4'), ('framecount', '<i8'), ('unixnano', '<i8'), ('pretriggerMean', '<f4'), ('residualStdDev', '<f4'), ('pulseMean', '<f4'), ('derivativeLike', '<f4'), ('filtValue', '<f4'), ('extraCoefs', '<f4', (2,))]

recrods of off files numpy arrays with dtypes, which contain may filed. The exact set of fields depends on the off file version as they are still under heavy development. The projector coefficients are stored in "pulseMean", "derivativeLike", "filtValue" and "extraCoefs". You can access 

Fields in OFF v3:
 - ``recordSamples`` - forward looking for when we implement varible length records, the actual number of samples used to calculate the coefficients
 - ``recordPreSamples`` - forward looking for when we implement varible length records, the actual number of pre-samples used to calculate the coefficients
 - ``framecount`` - a timestamp in units of DAQ frames (frame = one sample per detector, 0 is related to the start time of DASTARD or the DAQ system)
 - ``unixnano`` - a timetamp in nanoseconds since the epoch as measured by the computer clock (posix time)
 - ``pretriggerMean`` - average value of the pre-trigger samples (this may be removed in favor of calculating it from the pulse coefficients)
 - ``residualStdDev`` - the standard deviation of the residuals of the raw pulse - the reconstructed model pulse
 - ``pulseMean`` - the mean of the whole pulse record
 - ``derivativeLike`` - coefficient of the derivativeLike pulse shape
 - ``filtValue`` - coefficient of the optimal filter pulse shape
 - ``extraCoefs`` - zero or more additional pulse coefficients one for each other projector used, these generally model pulse shape variation vs energy and are found via SVD

Basic Analysis with a ``ChannelGroup``
--------------------------------------

Data analysis is generally done within the ``ChannelGroup`` class, by convention we call we store it in a variable ``data`` and 
name any particular channel ``ds``. There are convenience functions for many plotting needs, here we plot a histogram of ``filtValue``. 

.. testcode::

  from mass.off import ChannelGroup, getOffFileListFromOneFile, Channel, labelPeak, labelPeaks
  data = ChannelGroup(getOffFileListFromOneFile("../mass/off/data_for_test/20181205_BCDEFGHI/20181205_BCDEFGHI_chan1.off", maxChans=2))

I typically just label states alphabetically while taking data, but it can be convenient to alias them with some meaningful name for analysis.
Currently this must be done before you access any data from the OFF file.

.. testcode::

  data.experimentStateFile.aliasState("B", "Ne")
  data.experimentStateFile.aliasState("C", "W 1")
  data.experimentStateFile.aliasState("D", "Os")
  data.experimentStateFile.aliasState("E", "Ar")
  data.experimentStateFile.aliasState("F", "Re")
  data.experimentStateFile.aliasState("G", "W 2")
  data.experimentStateFile.aliasState("H", "CO2")
  data.experimentStateFile.aliasState("I", "Ir")

Then we can learn some basic cuts and get an overview of the data. 

.. testcode::

  data.learnResidualStdDevCut(plot=True)
  ds = data[1]
  ds.plotHist(np.arange(0,25000,10),"filtValue", coAddStates=False)

.. testcode::
  :hide:

  plt.savefig("img/hist1.png");plt.close() # opposite order of how they were plotted
  plt.savefig("img/residualStdDevCut.png");plt.close()


.. image:: img/residualStdDevCut.png
  :width: 45%

.. image:: img/hist1.png
  :width: 45%

Here we opened all the channels that have the same base filename, plus we opened the ``_experiment_state.txt`` that defines states. 
States provide a convenient way to seperate your data into different chunks by time, and a generally assigned during data aquistion, 
but you can always make a new ``_experiment_state.txt`` file to split things up a different way.

To calibrate the data we create a ``CalibrationPlan``, for now we do it manually on just one channel. See how we can call out which 
state or states a paricular line appears in. One a channel has a ``CalibrationPlan`` the recipe ``energyRough`` will be defined.

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

  ds.plotHist(np.arange(0, 4000, 1), "energyRough", coAddStates=False)

.. testcode::
  :hide:

  plt.savefig("img/hist2.png");plt.close()

.. image:: img/hist2.png
  :width: 45%

Now we use ``ds`` as a reference channel to use dynamitc time warping based alignment to create a matching calibration plan for each other channel.
Now we can make co-added ``energyRough`` plots. The left plot will be showing how the alignment algorithm works by identifying the same peaks in 
each channel, and the next will be a coadded energy plot. If ``alignToReferenceChannel`` complains, it often helps to increase the range used.

Also notice the coadded plot function is identical to the single channel function, just use it on ``data``. Many function work this way.

.. testcode::

  data.alignToReferenceChannel(referenceChannel=ds,
                              binEdges=np.arange(500, 20000, 4), attr="filtValue", _rethrow=True)
  aligner = data[3].aligner
  aligner.samePeaksPlot()
  data.plotHist(np.arange(0, 4000, 1), "energyRough", coAddStates=False)

.. testcode::
  :hide:

  plt.savefig("img/coadded_energy_rough_hist1.png");plt.close()
  plt.savefig("img/aligner.png");plt.close()


.. image:: img/aligner.png
  :width: 45%

.. image:: img/coadded_energy_rough_hist1.png
  :width: 45%

Now lets learn a correciton to remove some correlation between ``pretriggerMean`` and ``filtValue``. First we create a cut recipe to select only pulses in a certain energy 
range so hopefully the correction will work better in that range. 
So far, it is not super easy to combine two cuts, but we'll figure that out eventually, and you can just make a 3rd cut recipe to do so.
This will create a recipe for each channel called ``filtValueDC``, though the name can be adjusted with some arguments.
There are functions for phase corecction and time drift correction. Don't expect magic from these, they can only do so much.

Then we calibrate each channel, using ``filtValueDC`` as the input. This creates a recipe ``energy`` which is calibrated based on fits to each line in the ``CalibrationPlan``.

.. testcode::

  data.cutAdd("cutForLearnDC", lambda energyRough: np.logical_and(
      energyRough > 1000, energyRough < 3500), setDefault=False, _rethrow=True)
  data.learnDriftCorrection(uncorrectedName="filtValue", indicatorName="pretriggerMean", correctedName="filtValueDC", 
    states=["W 1", "W 2"], cutRecipeName="cutForLearnDC", _rethrow=True)
  data.calibrateFollowingPlan("filtValueDC", calibratedName="energy", dlo=10, dhi=10, approximate=False, _rethrow=True, overwriteRecipe=True)
  ds.diagnoseCalibration()

.. testcode::
  :hide:

  plt.savefig("img/diagnose.png");plt.close()

.. image:: img/diagnose.png
  :width: 95%

Then you will often want to fit some lines. 

.. testcode::

  data.linefit("W Ni-20", states=["W 1", "W 2"])

.. testcode::
  :hide:

  plt.savefig("img/linefit.png");plt.close()

.. image:: img/linefit.png
  :width: 45%


Data
----

``ChannelGroup`` is a dictionaries of channels. So you may want to loop over them. 

.. testcode::

  for ds in data.values(): # warning this will change the meaning of your existing ds variable
    print(ds)

.. testoutput::
  :options: +NORMALIZE_WHITESPACE

  Channel based on <OFF file> ../mass/off/data_for_test/20181205_BCDEFGHI/20181205_BCDEFGHI_chan1.off, 19445 records, 5 length basis

  Channel based on <OFF file> ../mass/off/data_for_test/20181205_BCDEFGHI/20181205_BCDEFGHI_chan3.off, 27369 records, 5 length basis



Bad channels
------------

Some channels will fail various steps of the analysis, or just not be very good. They will be marked bad. I'm also trying to develop some quality check algorithms.

.. testcode::

  results = data.qualityCheckLinefit("Ne H-Like 3p", positionToleranceAbsolute=2,
                                   worstAllowedFWHM=4.5, states="Ne", _rethrow=True,
                                   resolutionPlot=False)
  # all channels here actually pass, so lets pretend they dont
  ds.markBad("pretend failure")
  print(data.whyChanBad)
  ds.markGood()

.. testoutput::

  OrderedDict([(3, 'pretend failure')])

Recipes
-------

During everything in this tutorial, and most off style analsys usage, nothing has been written to disk, and very little is in memory.
This is imporant to avoid complicated tracking of state that lead to corruption of HDF5 files in mass, and to allow fast analysis of sub-sets 
of very large datasets. Most items, eg ``energy``, are computed lazily (on demand) based on a ``Recipe``. Here we can inspect the recipes of ``ds``.

Below I show how to add a recipe and inspect existing recipes.

.. testcode::
  
  ds.recipes.add("timeSquared", lambda framecount, relTimeSec: framecount*relTimeSecond) 
  ds.recipes.add("timePretrig", lambda timeSquared, pretriggerMean: timeSquared*pretriggerMean)
  print(ds.recipes)

.. testoutput::
  :options: +NORMALIZE_WHITESPACE

  RecipeBook: baseIngedients=recordSamples, recordPreSamples, framecount, unixnano, pretriggerMean, residualStdDev, pulseMean, derivativeLike, filtValue, extraCoefs, craftedIngredeints=relTimeSec, filtPhase, cutNone, cutResidualStdDev, energyRough, arbsInRefChannelUnits, cutForLearnDC, filtValueDC, energy, timeSquared, timePretrig

Linefit
-------

``X.linefit`` is a convenience method for quickly fitting a single lines. Here we show some of the options.

.. testcode::

  import lmfit
  # turn off the linear background if you want to later create a composite model, having multiple background functions messes up composite models
  ds.linefit("W Ni-20", states=["W 1", "W 2"], has_linear_background=False)

  # add tails and specify their parameters
  p = lmfit.Parameters()
  p.add("tail_frac_hi", value=0.01, min=0, max=1)
  p.add("tail_tau_hi", value=8, vary=False)
  p.add("tail_tau", value=8, vary=False)
  p.add("tail_frac_hi", value=0.04, min=0, max=1)
  ds.linefit("W Ni-20", states=["W 1", "W 2"], has_linear_background=False, has_tails=True, params_update=p)

.. testcode::
  :hide:

  plt.savefig("img/linefit_no_bg.png");plt.close()
  plt.savefig("img/linefit_tail_hi.png");plt.close() 

.. image:: img/linefit_no_bg.png
  :width: 45%

.. image:: img/linefit_tail_hi.png
  :width: 45%