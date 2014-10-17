# The HDF5 module lacks some features that we really want to use, specifically
# the ability to create groups, datasets, or attributes when they do NOT already
# exist, but simply to update them if they DO exist.

# This module contains "helper" methods to do that.
#
# Joe Fowler, NIST
# October 2014
#

module HDF5Helpers

export
h5file_update,
g_create_or_open,
ds_update,
attr_update

using HDF5

# Open an existing file for appending, or create a new one
# for writing.

function h5file_update(filename::String)
    try
        isfile(filename) && return h5open(filename, "r+")
    catch
    end
    h5file = h5open(filename, "w")
end


# Create a new or open an existing group within an HDF5 object
# If you can figure out a native syntax that handles both cases,
# then we'd prefer to use it.
function g_create_or_open(parent::Union(HDF5File,HDF5Group),
                          name::Union(UTF8String,ASCIIString))
    if exists(parent, name)
        return parent[name]
    end
    g_create(parent, name)
end



# Create a new or update an existing dataset within an HDF5 object
# If you can figure out a native syntax that handles both cases,
# then we'd prefer to use it.
function ds_update(parent::Union(HDF5File,HDF5Group),
                   name::Union(UTF8String,ASCIIString),
                   value::Any)
    if exists(parent, name)
        o_delete(parent, name)
    end
    parent[name] = value
end

# Create a new or update an existing complex-valued dataset within an HDF5 object
# If you can figure out a native syntax that handles both cases,
# then we'd prefer to use it.
function ds_update(parent::Union(HDF5File,HDF5Group),
                   name::Union(UTF8String,ASCIIString),
                   value::Vector{Complex128})
    if exists(parent, name)
        o_delete(parent, name)
    end
    val = hcat(real(value), imag(value))
    parent[name] = val
    attr_update(parent[name], "complex", "true")
    parent[name]
end


# Create a new or update an existing attribute within an HDF5 object
# If you can figure out a native syntax that handles both cases,
# then we'd prefer to use it.
function attr_update(parent::Union(HDF5File,HDF5Group,HDF5Dataset),
                   name::Union(UTF8String,ASCIIString),
                   value::Any)
    if exists(attrs(parent), name)
        # Don't do anything if the existing attribute is already correct
        a_read(parent, name) == value && return value
        a_delete(parent, name)
    end
    attrs(parent)[name] = value
end


end # module