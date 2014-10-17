module MicrocalFiles

export
LJHHeader,
LJHFile,
LJHRewind,
fileRecords,
fileData,
hdf5_name_from_ljh_name

include("LJHFile.jl")



# Given an LJH file name, return the HDF5 name
# Generally, /x/y/z/data_taken_chan1.ljh becomes /x/y/z/data_taken_mass.hdf5
function hdf5_name_from_ljh_name(ljhname::String, namesuffix::String="mass")
    dir = dirname(ljhname)
    base = basename(ljhname)
    path,suffix = splitext(ljhname)
    m = match(r"_chan\d+", path)
    path = string(path[1:m.offset-1], "_$namesuffix.hdf5")
end

end # module
