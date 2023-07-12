# Cuts

In mass, the "cuts" objects handle two distinct concepts:
* **Boolean cuts** handle yes/no decisions about whether a certain triggered record is "valid", high enough quality to be used for analysis.
* **Categories**--also known as _categorical cuts_--divide records into multiple, named groups of records. These are used to handle a series of "experiment states", or any other mutually exclusive categories.

### Boolean cuts

If boolean cuts have already been computed based on data-quality tests, then the basic usage is as simples as using `ds.good()` to index any vectors that have a length of `ds.nPulses`. Like this:
```python
ds = data.channel[13]
g = ds.good()
plt.clf()
plt.hist(ds.p_filt_value[g], 1000, histtype="step")
```

Here are some more complicated examples:

```python
# The following return boolean vectors, with one value per pulse:
ds.good()  # works as normal; returns True is good by *all* boolean crieteria.
ds.good("pretrigger_rms")  # returns True for pulses that are good by the "pretrigger_rms" criterion.
            # Other boolean criteria are ignored

# Change the contents of the "pretrigger_rms" cut field to match the values in boolvec
# (a boolean vector with one entry per pulse).
ds.cuts.cut("pretrigger_rms", boolvec)

# There are default boolean cut fields. To see the currently defined fields:
In [7]: data.boolean_cut_desc
Out[7]:
array([('pretrigger_rms', 1L), ('pretrigger_mean', 2L),
       ('pretrigger_mean_departure_from_median', 4L), ('peak_time_ms', 8L),
       ('rise_time_ms', 16L), ('postpeak_deriv', 32L),
       ('pulse_average', 64L), ('min_value', 128L),
       ('timestamp_sec', 256L), ('timestamp_diff_sec', 512L),
       ('peak_value', 1024L), ('energy', 2048L), ('timing', 4096L),
       ('p_filt_phase', 8192L), ('smart_cuts', 16384L), ('', 0L), ('', 0L),
       ('', 0L), ('', 0L), ('', 0L), ('', 0L), ('', 0L), ('', 0L),
       ('', 0L), ('', 0L), ('', 0L), ('', 0L), ('', 0L), ('', 0L),
       ('', 0L), ('', 0L), ('', 0L)],
      dtype=[('name', 'S64'), ('mask', '<u4')])

# Add a new boolean cut field and set its values
data.register_boolean_cut_fields("gut_feeling")
failscut = ds.p_gut_feeling_figure_of_merit[:] > 3500
ds.cuts.cut("gut_feeling", failscut)
```
Careful! You pass `True` to cut (to mark the pulse bad), not `False`!

Internally, cuts have code numbers, but you shouldn't ever have to use them. The name interface is a lot clearer.


### Categorical cuts
Categorical cuts are used when you want to label your pulses as belonging to one of several, mutually exclusive groups (categories).

There is by default only one categorical cut field `calibration`; it has two categories `in`
and `out`. By default every pulse is in `in`. The functions `ds.drift_correct`,
`ds.phase_correct_2014`, `ds.phase_correct`, and `ds.calibrate` all use the category `in` by
default. If you want to use only some pulses for calibration, you can do the following for
each dataset:

```python
# Assume that use_for_calibration is a boolean numpy array of length ds.nPulses.
ds.cuts.cut_categorical("calibration", {"in": use_for_calibration,
                                        "out": ~use_for_calibration})
```

For each categorical cut, there is always an implied category "uncategorized", so that even
a pulse assigned to no category does actually have a category.

Suppose you want to register a new categorical cut field. For example, in a pump-probe
experiment you may want to cut based on optical pump status. First, you would register
the name of the cut and the allowed category values:

```python
data.register_categorical_cut_field("pump",["pumped","unpumped"])
```

This registers a new bit field, two bits wide, to represents the three categories
"pumped", "unpumped", and "uncategorized". By default all pulses start out as "uncategorized". (The category "calibration" is weird because it's default category is "in").

To assign categories, we do

```python
ds.cuts.cut_categorical("pump",{"pumped":pumped_bool,
                                "unpumped":unpumped_bool})
```
Here, `pumped_bool` and `unpumped_bool` are 1 per pulse vectors of booleans,
`True` in `pumped_bool` for the pulse records that are pumped. `True` in `unpumped_bool` for the pulse records that are not pumped. If any pulses have `True` in both
`pumped_bool` and `unpumped_bool`, the function will raise an error. If any pulses have
`False` in both vectors, those pulses will be assigned to "uncategorized".

Now use the category. The first function returns True for pulses that pass all boolean cut fields AND is in the category "pumped". The second does drift correct using only `pump="pumped"` pulses (instead of the default of `calibration="in"`).

```python
ds.good(pump="pumped")
data.drift_correct(forceNew=False, category={"pump":"pumped"})
```

If you want to remove a categorical cut (perhaps to change the cut's list of categories),
or one or more boolean cuts, you can use the following:

```python
data.unregister_categorical_cut_field("pump")
data.unregister_boolean_cut_fields(["smelly", "malodorous"])
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
