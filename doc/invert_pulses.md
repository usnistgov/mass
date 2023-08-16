## Invert Pulses

If you have a channel that was wired up backwards, so that pulses are
negative-going, then you can fix this in analysis by saying:
```
for channum in [5,17,23,25]:
    ds = data.channel[channum]
    ds.invert_data = True
```
This will mean that all future reads of raw data will get the unit16-
inverse of the data actually recorded to the LJH file.
