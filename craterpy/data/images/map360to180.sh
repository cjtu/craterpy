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