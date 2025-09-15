# Sample data imagery

Files were drawn from the [USGS Astropedia](https://astrogeology.usgs.gov/search) site in Sep 2025.

To produce low resolution context images, the sample JPEG imagery of various planetary bodies were converted to `.tif` files using [GDAL](https://gdal.org/).

```sh
#!/bin/bash
#
# Convert JPEG to single-band grayscale GeoTIFF (body_code: Mercury=199, Moon=300, etc)
# Note: set -a_ullr correctly in the script for global maps (-180,180) or (0,360)
# Usage: jpeg2geotiff <in.jpg> <out.tif> <body_code>

if [ "$#" -ne 3 ]; then
    echo "Usage: jpeg2geotiff <in.jpg> <out.tif> <body_code>"
    exit 1
fi

# Get number of bands using gdalinfo, convert RGB to grayscale if necessary
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

# Add projection. NOTE: Maps from 0-360 must change -a_ullr 0 90 360 -90 
gdal_translate \
    -co "COMPRESS=LZMA" \
    -co "NUM_THREADS=12" \
    -a_srs "IAU_2015:${3}00" \
    -a_ullr -180 90 180 -90 \
    "$input" "$2"

# Clean up temporary file if it exists
[ -f temp.tif ] && rm temp.tif
```

Then to convert the 0,360 files to -180,180, the following script was applied:

```sh
#!/bin/bash
#
# Convert 0-360 GeoTIFF to -180-180 keeping same CRS
# Usage: map360to180.sh <in.jpg> <out.tif>
# See: https://gis.stackexchange.com/a/465906

if [ "$#" -ne 2 ]; then
    echo "Usage: map360to180.sh <in.tif> <out.tif>"
    exit 1
fi

IN=$1
OUT=$2
RASTER_XSIZE=$(gdalinfo "$IN" | grep "Size is" | awk '{print $3}')
CX=$(echo "${RASTER_XSIZE//,} / 2" | bc)
echo $RASTER_XSIZE
gdal_translate -srcwin 0 0 $CX $CX -a_ullr 0 90 180 -90 $IN tmp_right.vrt
gdal_translate -srcwin $CX 0 $CX $CX -a_ullr -180 90 0 -90 $IN tmp_left.vrt
gdalbuildvrt tmp.vrt tmp_left.vrt tmp_right.vrt
gdal_translate tmp.vrt $OUT 

# Clean up temporary file if it exists
[ -f tmp.vrt ] && rm tmp.vrt
[ -f tmp_right.vrt ] && rm tmp_right.vrt
[ -f tmp_left.vrt ] && rm tmp_left.vrt
```