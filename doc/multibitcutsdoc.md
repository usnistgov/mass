In Sept 2015 Young introduced a new cut API that is both more functional, and faster in many cases than the previous API.

There are two types of cut fields, `boolean` and `categorical`.

#### Boolean 
```
ds.good() # works as normal returning True for pulses that are good by every boolean critera
ds.good("pretrigger_rms") # returns True for pulses that are good by the "pretrigger_rms" critera, ignoring all other boolean fields
ds.cuts.cut("pretrigger_rms",boolvec) # changes the contents of the "pretrigger_rms" cut field to match the values in boolvec (on entry per pulse)

# the default boolean cut fields are
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
      
# if you want to add a new boolean cut field
data.register_boolean_cut_fields("gut_feeling")
ds.cuts.cut("gut_feeling", ds.p_gut_feeling_figure_of_merit[:] > 3500) #note pass true to cut, aka pass true to mark the pulse bad
```

You should never need to look up or use a cut number again, just always use the name for the field.

#### Categorical 
Categorical cuts are used when you want to evaluate your pulses as different groups (categories). The simplest example is the only default categorical cut field `calibration` which has two categories `in` and `out`. 
By default every pulse is in `in`. The functions `ds.drift_correct`, `ds.phase_correct_2014`, and `ds.calibrate` all use the category `in` by default. So if you want to use only some pulses for calibration, you can do this for each dataset.

```
ds.cuts.cut_categorical("calibration",{"in":use_for_calibration,
                                    "out":~use_for_calibration})
```

If you want to register a new categorical cut, for example in a pump-probe experiment you may want to cut based on delay stage position. You would do
`data.register_categorical_cut_field("pump",["pumped",unpumped"])`, this registers a new 2 bit field to represent 3 categories "pumped", "unpumped", and "uncategorized". By default all pulses are in "uncategorized" (calibration is weird because it's default category is "in"). To apply the cut we do
`ds.cuts.cut_categorical("pump",{"pumped":pumped_bool, "unpumped":unpumped_bool})`. `pumped_bool` is a 1 per pulse vector of bools, `True` for the ones that are pumped. If any pulses have `True` in both `pumped_bool` and `unpumped_bool`
it will raise an error. If any pulses have `False` in both, those pulses will be assigned to "uncategorized".

```
ds.good(pump="pumped") # returns True for pulses that pass all boolean cut fields AND is in category pumped
data.drift_correct(forceNew=False,category={"pump":"pumped"}) # does drift correct with only "pump":"pumped" pulses instead of "calibration":"in"
```

There is an alternate API that may be more convenient in some cases. Imagine we have an experiment with a delay stage that was in many positions thruout the experiment.

```
data.register_categorical_cut_field("delay",["-150mm","-100mm","-50mm",...,"200mm"])
categories = data.cut_field_categories("delay") # gets a dict of category name ("-150mm") to category label (1)
labels = np.zeros(ds.nPulses, dtype=np.int64) # note 0 is always the default category, unless otherwise specified named "uncategorized"
for i in range(ds.nPulses):
    delay_stage_pos = get_delay_stage_pos(i)
    labels[i]=categories[delay_stage_pos] # labels is a one per pulse vector with integer valued labels, each integer corresponds to a category name in the dictionary categories
    ds.cuts.cut("delay", labels)
```



