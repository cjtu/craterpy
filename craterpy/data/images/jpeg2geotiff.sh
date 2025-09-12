#!/bin/bash
#
# Convert JPEG to single-band grayscale GeoTIFF
# Usage: jpeg2geotiff <in.jpg> <out.tif> <body_code>

if [ "$#" -ne 3 ]; then
    echo "Usage: jpeg2geotiff <in.jpg> <out.tif> <body_code>"
    exit 1
fi

# Get number of bands using gdalinfo
num_bands=$(gdalinfo "$1" | grep -c "Band ")

if [ "$num_bands" -eq 1 ]; then
    input="$1"
else
    echo "Converting $num_bands bands to grayscale..."
    # Convert to grayscale using ITU-R BT.601 weights
    gdal_calc.py \
        -A "$1" --A_band=1 \
        -B "$1" --B_band=2 \
        -C "$1" --C_band=3 \
        --outfile="temp.tif" \
        --calc="0.299*A + 0.587*B + 0.114*C" \
        --type=Byte \
        --NoDataValue=0
    input="temp.tif"
fi

# Add projection and compress  # -a_ullr -0 90 360 -90 \
gdal_translate \
    -co "COMPRESS=LZMA" \
    -co "NUM_THREADS=12" \
    -a_srs "IAU_2015:${3}00" \
    -a_ullr -180 90 180 -90 \
    "$input" "$2"

# Clean up temporary file if it exists
[ -f temp.tif ] && rm temp.tif