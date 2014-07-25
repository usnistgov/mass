export
LJHHeader,
LJHFile,
LJHRewind,
fileRecords,
fileData

###############################################################################
# DATA STRUCTURES
###############################################################################

# LJH file header information (keeping only the necessary elements)
immutable LJHHeader
    filename         ::String
    npresamples      ::Int64
    nsamples         ::Int64
    timebase         ::Float64
    timestampOffset  ::Float64
    date             ::String
    headerSize       ::Int64
    channum          ::Uint16
end



const LJH_RECORD_HDR_SIZE=6
const LJH_DATA_SIZE=sizeof(Int16)

# LJH file abstraction.
# Constructor reads/parses header, then opens file and points stream to
# the first non-header bytes.
immutable LJHFile
    name             ::String        # filename
    str              ::IOStream      # IOStream to read from LJH file
    header           ::LJHHeader     # LJH file header data
    nrec             ::Int64         # number of (pulse) records in file
    dt               ::Float64       # sample spacing (microseconds)
    npre             ::Int64         # npresamples
    nsamp            ::Int64         # nsamples per record
    reclength        ::Int64         # record length (bytes)
    channum          ::Uint16        # channel number

    function LJHFile(name::ASCIIString)
        hd = readLJHHeader(name)
        dt = hd.timebase
        pre = hd.npresamples
        tot = hd.nsamples
        channum = hd.channum
        datalen = stat(name).size - hd.headerSize
        reclen = LJH_RECORD_HDR_SIZE+LJH_DATA_SIZE*tot
        nrec = div(datalen,reclen)

        str = open(name)
        seek(str,hd.headerSize)
        new(name, str, hd, nrec, dt, pre, tot, reclen, channum)
    end
end


###############################################################################
# FILE-READING FUNCTIONS
###############################################################################

# Get pertinent information from LJH file header and return it as LJHHeader
# filename = path to the LJH file
# Returns: new LJHHeader object.
function readLJHHeader(filename::String)
    str=open(filename)
    labels={"base"   =>"Timebase:",
            "date"   =>"Date:",
            "date1"  =>"File First Record Time:",
            "end"    =>"#End of Header",
            "offset" =>"Timestamp offset (s):",
            "pre"    =>"Presamples: ",
            "tot"    =>"Total Samples: ",
            "channum"=>"Channel: "}
    nlines=0
    maxnlines=100
    date = "unknown" # If header standard for date labels changes, we don't want a hard error

    # Read channel # from the file name, then update that result from the header, if it exists.
    channum = uint16(-1)
    m = match(r"_chan\d+", filename)
    channum = uint16(m.match[6:end])

    while nlines<maxnlines
        line=readline(str)
        nlines+=1
        if beginswith(line,labels["end"])
            headerSize = position(str)
            close(str)
            return(LJHHeader(filename,npresamples,nsamples,
                             timebase,timestampOffset,date,headerSize,channum))
        elseif beginswith(line,labels["base"])
            timebase = float64(line[1+length(labels["base"]):end])
        elseif beginswith(line,labels["date"]) # Old LJH files
            date = line[7:end-2]
        elseif beginswith(line,labels["date1"])# Newer LJH files
            date = line[25:end-2]
        elseif beginswith(line,labels["channum"])# Newer LJH files
            channum = uint16(line[10:end])
        elseif beginswith(line,labels["offset"])
            timestampOffset = float64(line[1+length(labels["offset"]):end])
        elseif beginswith(line,labels["pre"])
            npresamples = int64(line[1+length(labels["pre"]):end])
        elseif beginswith(line,labels["tot"])
            nsamples = int64(line[1+length(labels["tot"]):end])
        end
    end
    error("read_LJH_header: where's '$(labels["end"])' ?")
end



# Read the next nrec records and for each return time and samples
# (error if eof occurs or insufficient space in times or data)
function fileRecords(f::LJHFile, nrec::Integer,
                     times::Vector{Uint64}, data::Matrix{Uint16})
    #assert(nrec <= min(length(times),size(data,2)) && size(data,1)==f.nsamp)
    for i=1:nrec
        times[i] = recordTime(read(f.str, Uint8, LJH_RECORD_HDR_SIZE))
        data[:,i] = read(f.str, Uint16, f.nsamp)
    end
end


function LJHRewind(f::LJHFile)
    seek(f.str, f.header.headerSize)
end


# Read specific record numbers and for each return time and samples;
# restore file position (error if eof occurs or insufficient space in times or data)
function fileRecords(f::LJHFile, recIndices::Vector{Int},
                     times::Vector{Uint64}, data::Matrix{Uint16})
    savedposition = position(f.str)
    for i = 1:length(recIndices)
        seek(f.str, f.header.headerSize+(recIndices[i]-1)*f.reclength)
        times[i] = recordTime(read(f.str, Uint8, LJH_RECORD_HDR_SIZE))
        data[:,i] = read(f.str, Uint16, f.nsamp)
    end
    seek(f.str, savedposition)
end



# From LJH file, return all data samples as single vector. Drop the timestamps.
function fileData(filename::String)
    ljh = LJHFile(filename)
    time = Array(Uint64, ljh.nrec)
    data = Array(Uint16, ljh.nsamp, ljh.nrec)
    fileRecords(ljh, ljh.nrec, time, data)
    close(ljh.str)
    vec(data)
end



# Decode the record time packed into the 6-byte record header (LJH version 2.1.0)
# Returns: time in microseconds, relative to the file's header.timestampOffset
function recordTime(header::Vector{Uint8})
    frac = uint64(header[1])
    ms = uint64(header[3]) |
         (uint64(header[4])<<8) |
         (uint64(header[5])<<16) |
         (uint64(header[6])<<24)
    return ms*1000 + frac*4
end



# Given a time, return a valid 6-byte LJH record header.
# Allocation-free version
# t = record start time in microseconds since the file's header.timestampOffset
# header = pre-allocated 6-byte vector for storing the result.
function recordHeader(t::Uint64, header::Vector{Uint8})
    frac = div(t%1000,4)
    ms = uint32(div(t,1000))
    header[1] = uint8(frac)
    header[2] = uint8(0)
    header[3] = uint8(0xFF & ms)
    header[4] = uint8(0xFF & (ms >> 8))
    header[5] = uint8(0xFF & (ms >> 16))
    header[6] = uint8(0xFF & (ms >> 24))
    return header
end



# Given a time, return a valid 6-byte LJH record header (LJH version 2.1.)
# Note that record headers have only 4 us least-bit resolution.
# t = record start time in microseconds since the file's header.timestampOffset
# Returns: newly allocated 6-byte vector containing record header.
recordHeader(t::Uint64) = recordHeader(t, Array(Uint8, LJH_RECORD_HDR_SIZE))
