Many of `mass.core.channel_group.TESGroup` methods set `mass.core.channel.MicrocalDataSet` bad when it catches any exception while it processes with it. This is done is by calling `mass.core.channel_group.TESGroup.set_chan_bad` And this information is also stored in a HDF5 file so that when data is loaded again, it will retain the same bad channel list.

```python3
data.set_chan_bad(13, [1, 9], 'failed p_filt_value_dc calibration')

# bad channel information is stored in a python dictionary.
for chan_num in [1, 9, 13]:
    assert 'failed p_filt_value_dc calibration' in data._bad_channums

# bad channel information is also stored in a hdf5 file.
for chan_num in [1, 9, 13]:
    hdf5_group = data.hdf5_file['chan{0:d}'.format(chan_num)]
    assert 'failed p_filt_value_dc calibration' in hdf5_group.attrs['why_bad']
```