import h5py
from . import channel_group
from . import channel
import os
def hdf5jl_name_from_ljh_name(ljh_name):
    b,ext = os.path.splitext(ljh_name)
    return b+"_jl.hdf5"

def make_or_get_master_hdf5_from_julia_hdf5_file(hdf5_filenames=None, forceNew=False, require_clean_exit=True):
    h5master_fname = channel_group._generate_hdf5_filename(hdf5_filenames[0])
    if os.path.isfile(h5master_fname):
        if forceNew:
            os.remove(h5master_fname)
        else:
            print("RESUING EXISTING MASTER HDF5 FILE, %s"%h5master_fname)
            return h5master_fname

    with h5py.File(h5master_fname,"a") as master_hdf5_file:
        with h5py.File(hdf5_filenames[0],"r") as single_channel_file:
            # put the data where python mass expects it
            master_hdf5_file.attrs["nsamples"]=single_channel_file["samples_per_record"].value
            master_hdf5_file.attrs["npresamples"]=single_channel_file["pretrig_nsamples"].value
            master_hdf5_file.attrs["frametime"]=single_channel_file["filter/frametime"].value
        for h5fname in hdf5_filenames:
            i = h5fname.find("_chan")
            channum = int(h5fname[i+5:-8])
            try:
                with h5py.File(h5fname,"r+") as single_channel_file:
                    if ("clean_exit_posix_timestamp_s" in single_channel_file or not require_clean_exit) and len(single_channel_file["filt_value"][:])>1:
                        if not "channum" in single_channel_file.attrs.keys(): single_channel_file.attrs["channum"]=channum
                        if not "npulses" in single_channel_file.attrs.keys(): single_channel_file.attrs["npulses"]=len(single_channel_file["filt_value"])
                        single_channel_file.attrs["filename"] = h5fname
                master_hdf5_file["chan%i"%channum] = h5py.ExternalLink(h5fname, "/")
            except KeyError:
                print("failed to load chan %d hdf5 only"%channum)

    return h5master_fname

class TESGroupHDF5(channel_group.TESGroup):
    def __init__(self, h5master_fname):
        self.hdf5_file = h5py.File(h5master_fname,"a")
        self.nPresamples = self.hdf5_file.attrs["npresamples"]

        self.cut_field_desc_init()

        dset_list = []
        for key in self.hdf5_file.keys():
            if not key.startswith("chan"):
                continue
            grp = self.hdf5_file[key]
            # print(grp)
            # print(grp.keys())
            # print grp.attrs["channum"]
            pulserec_dict = {"nSamples":self.hdf5_file.attrs["nsamples"],
                             "nPresamples":self.nPresamples,
                             "nPulses":grp.attrs["npulses"],
                             "timebase":self.hdf5_file.attrs["frametime"],
                             "channum":grp.attrs["channum"],
                             "timestamp_offset":0,
                             "filename":grp.file.filename}
            dset_list.append(channel.MicrocalDataSet(pulserec_dict, tes_group=self, hdf5_group=grp))

        self.datasets = tuple(dset_list)
        self._bad_channums = {}
        self._setup_channels_list()
        self.fix_timestamps()


    def fix_timestamps(self):
        """Mass expects p_timestamp to have units of seconds and be a float, sometimes we save microsecond units ints.
        This is a way to give matter what it expects."""
        for ds in self:
            grp = ds.hdf5_group
            if "timestamp_posix_usec" in grp:
                ds.p_timestamp = grp["timestamp_posix_usec"][:]*1e-6
