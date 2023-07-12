# Cuts

In mass, the "cuts" objects handle two distinct concepts:
* [**Boolean cuts**](#boolean-cuts) handle yes/no decisions about whether a certain triggered record is "valid", high enough quality to be used for analysis.
* [**Categories**](#categorical-cuts)--also known as _categorical cuts_--divide records into multiple, named groups of records. These are used to handle a series of "experiment states", or any other mutually exclusive categories.

## Boolean cuts

If boolean cuts have already been computed based on data-quality tests, then the basic usage is as simples as using the results of calling `ds.good()` as an index into any vectors that have a length of `ds.nPulses`. Like this:
```python
ds = data.channel[13]
g = ds.good()
plt.clf()
plt.hist(ds.p_filt_value[g], 1000, histtype="step")
```

Here are some more complicated examples:

```python
# The following return boolean vectors, with one value per pulse:
ds.good()  # works as normal; returns True when pulse is good by *all* boolean crieteria.
ds.good("pretrigger_rms")  # returns True for pulses that are good by the "pretrigger_rms" criterion;
                           # other boolean criteria are ignored.

# Replace the contents of the "pretrigger_rms" cut field with the values in boolvec
# (a boolean vector with one entry per pulse).
ds.cuts.cut("pretrigger_rms", boolvec)

# Several boolean cut fields are pre-defined. To see the currently defined fields:
In [17]: data.boolean_cut_desc
Out[17]:
                 name                  |               mask
---------------------------------------+----------------------------------
            pretrigger_rms             | 00000000000000000000000000000001
            pretrigger_mean            | 00000000000000000000000000000010
 pretrigger_mean_departure_from_median | 00000000000000000000000000000100
             peak_time_ms              | 00000000000000000000000000001000
             rise_time_ms              | 00000000000000000000000000010000
            postpeak_deriv             | 00000000000000000000000000100000
             pulse_average             | 00000000000000000000000001000000
               min_value               | 00000000000000000000000010000000
             timestamp_sec             | 00000000000000000000000100000000
          timestamp_diff_sec           | 00000000000000000000001000000000
           rowcount_diff_sec           | 00000000000000000000010000000000
              peak_value               | 00000000000000000000100000000000
                energy                 | 00000000000000000001000000000000
                timing                 | 00000000000000000010000000000000
             p_filt_phase              | 00000000000000000100000000000000
              smart_cuts               | 00000000000000001000000000000000

# Add a new boolean cut field and set its values
data.register_boolean_cut_fields("hunch")
failscut = ds.p_some_quantity_that_should_be_small[:] > 3500
ds.cuts.cut("hunch", failscut)
```
Careful! You must pass `True` to cut a record, not `False`! That's because the function "cut" wants to know which
pulses ought to be cut.

Internally, cuts have code numbers, but you shouldn't ever have to use them. The name interface is a lot clearer.


## Categorical cuts
Categorical cuts are used when you want to label your pulses as belonging to one of several, mutually exclusive groups (categories).

There are two categorical cut fields, or categorizations, by default. More can be added, of course.
* `calibration` is used to signify which pulses are to be used for calibration. It has two
  categories, `in` and `out`. By default, every pulse is in `in`. In addition to calibration,
  the functions `ds.drift_correct`,
  `ds.phase_correct_2014`, `ds.phase_correct`, and `ds.calibrate` all use the category `calibration="in"` by
  default.
* `state` is built from the *experiment state file*, and it has as many categories as there are entries in
  that file.

Here's how you can see what categorizations and category labels already exist. The first example shows
all the categorizations ("field") and the available category labels in each. The second returns a dictionary
mapping each label to its internal code number for the specified categorical cut (here, "state").

```python
In [12]: data.cut_category_list
Out[12]:
    field    |   category    | code
-------------+---------------+------
 calibration |      in       |  0
 calibration |      out      |  1
    state    | uncategorized |  0
    state    |       A       |  1
    state    |       B       |  2
    state    |       C       |  3
    state    |       D       |  4
    state    |       E       |  5
    state    |       F       |  6

In [16]: data.cut_field_categories("state")
Out[16]:
{'uncategorized': 0,
 'A': 1,
 'B': 2,
 'C': 3,
 'D': 4,
 'E': 5,
 'F': 6}
```

If you want to use only some pulses for calibration, you can do the following for
each dataset:

```python
# Assume that use_for_calibration is a boolean numpy array of length ds.nPulses.
ds.cuts.cut_categorical("calibration", {"in": use_for_calibration,
                                        "out": ~use_for_calibration})
```

For each categorical cut other than "calibration", there is always an implicit category "uncategorized".
In this way, even a pulse not assigned to any category does actually have a category.

Suppose you want to register a new categorical cut field. For example, in a pump-probe
experiment you may want to cut based on optical pump status. First, you would register
the name of the cut and the allowed category values:

```python
data.register_categorical_cut_field("pump",["pumped","unpumped"])
```

This registers a new bit field, two bits wide, to represents the three categories
"pumped", "unpumped", and "uncategorized". By default all pulses start out as "uncategorized". (The category "calibration" is weird because it's default category is "in").

#### Assigning categories to the pulse records

To assign categories, we have two choices. One is to use a set of boolean vectors, each of length `ds.nPulses`, and one boolean per category. The [second (see below)](#alternate-api-for-categorial-cuts) uses a single vector of integers (also of length `ds.nPulses`), giving the category numbers for each pulse record. The boolean approach looks like this:

```python
ds.cuts.cut_categorical("pump",{"pumped":pumped_bool,
                                "unpumped":unpumped_bool})
```
Here, `pumped_bool` and `unpumped_bool` are vectors of booleans (one value per record), which is set
`True` in `pumped_bool` for the pulse records that are pumped and `True` in `unpumped_bool` for the pulse records that are not pumped.
If any pulses have `True` in both `pumped_bool` and `unpumped_bool`, the function will raise an error (pulses cannot be in more than one
category). Any pulses that have `False` in both vectors will be assigned to "uncategorized".

Now you can use the category. The first function returns True for pulses that pass all boolean cut fields AND is in the category "pumped". The second does drift correct using only `pump="pumped"` pulses (instead of the default of `calibration="in"`).

```python
ds.good(pump="pumped")
data.drift_correct(forceNew=False, category={"pump":"pumped"})
```

#### Alternate API for categorial cuts

There is an alternate API that may be more convenient in some cases. Imagine we
have an experiment with a delay stage that was in many positions throughout the
experiment.

```python
# Register a categorical cut field named "delay", and also get a dict
# that maps category label (e.g., "-150mm") to category code (1).

data.register_categorical_cut_field("delay", ["-150mm", "-100mm", "-50mm", "0mm", "50mm"])
categories = data.cut_field_categories("delay")

# category_codes is a one per pulse vector with integer valued codes, each integer
# corresponds to a category label in the dictionary categories.
# The value 0 is the default, indicating "uncategorized".

category_codes = np.zeros(ds.nPulses, dtype=np.uint32)
for i in range(ds.nPulses):
    delay_stage_pos = get_delay_stage_pos(i)
    category_codes[i] = categories[delay_stage_pos]

ds.cuts.cut("delay", category_codes)
```

#### Removing ("unregistering") a boolean or categorical cut

If you want to remove a categorical cut,
or one or more boolean cuts, you can unregister like this:

```python
data.unregister_categorical_cut_field("pump")
data.unregister_boolean_cut_fields("hunch")   # can unregister one at a time....
data.unregister_boolean_cut_fields(["smelly", "malodorous"])  # ...or a sequence of more than one
```

If you want to *change* the allowed set of categories in a categorical cut, you must remove it in this way and then register it again
with the updated set of category names.