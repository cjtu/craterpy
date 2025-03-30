---
title: 'Craterpy: Impact crater data science in Python.'
tags:
  - Python
  - planetary science
  - data analysis
  - spatial analysis
authors:
- name: Christian J. Tai Udovicic^[corresponding author]
  orcid: 0000-0001-9972-1534
  affiliation: "1"
- name: Ari Essunfeld 
  orcid: 0000-0001-8689-0734
  affiliation: "1"
- name: Emily S. Costello
  orcid: 0000-0001-7939-9867
  affiliation: "1"
affiliations:
- name: Hawaiʻi Institute of Geophysics and Planetology, School of Ocean and Earth Science and Technology, University of Hawaiʻi at Manoa, Honolulu, HI, United States of America
  index: 1
date: 30 March 2025
bibliography: paper.bib
---

# Summary

Impact craters and their associated annular and rayed ejecta pattern are central to the study of planetary geology. In previous work, we presented a method for understanding the evolution of the lunar surface over time though the extraction of image data associated with impact craters [@TaiUdovicic2020]. New databases can exceed 10$^8$ craters, requiring an efficient set of tools to extract and store data or summary statistics associated with these craters and their ejecta. `Craterpy` was written to facilitate impact crater data analysis by producing and managing regions of interest associated with impact craters, extracting data and summary statistics from associated spectral image data, and proving a clean and modular API for users to quickly generate spatial crater data and plug it into their preferred analysis tools. 

# Statement of need

Most crater data analysis tasks require the conversion of lists of crater locations to precise regions of interest with geographic metadata specific to the planetary body of interest, which can be time consuming and error-prone for non-specialists on large lists of craters. Although the python geospatial stack is maturing, many tools support only Earth-based projection information, leading to image and shapefile offsets when working with planetary data. `Craterpy` facilitates the production and management of planetary shapefiles and also facilitates extraction of planetary image data associated with impact craters for ease of use in data analytical tasks. It provides a class-based user-friendly interface for defining impact crater regions of interest (ROIs) on an input list of crater locations and extracting data from supplied raster datasets associated with each region. `Craterpy` is built to be modular and to plugin to tools users may already be familiar with (e.g., `geopandas` and easy export to GIS shapefile formats). The toolkit is tested on multiple planetary bodies to ensure spatially accurate data extraction and statistics calculations on planetary raster image data. `Craterpy` was designed to be used by planetary scientists and has been used in multiple scientific publications [@TaiUdovicic:2020;@Chertok:2022] and has been presented an an international lunar science conference [@TaiUdovicic:2024].

# Acknowledgements

This work was supported by NASA Discovery Data Analysis grant to E.S.C.

We would like to thank our users and community for their invaluable feedback and contributions over the years.

# References