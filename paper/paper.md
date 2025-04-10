---
title: 'Craterpy: Impact crater data science in Python.'
tags:
  - Python
  - planetary science
  - data analysis
  - spatial analysis
authors:
- given-names: Christian J.
  surname: Tai Udovicic
  orcid: 0000-0001-9972-1534
  affiliation: "1"
  corresponding: true
- name: Ari Essunfeld 
  orcid: 0000-0001-8689-0734
  affiliation: "1"
- name: Emily S. Costello
  orcid: 0000-0001-7939-9867
  affiliation: "1"
affiliations:
- name: Hawaiʻi Institute of Geophysics and Planetology, School of Ocean and Earth Science and Technology, University of Hawaiʻi at Manoa, Honolulu, HI, United States of America
  index: 1
date: 9 April 2025
bibliography: paper.bib
---

# Summary

Impact craters and their associated ejecta patterns are useful for the study of planetary geology since impacts excavate material from below the surface and deposit it in a geologic instant before being subject to erosion and degradation over time. For example, we can understand the rate that the surface of the Moon evolves over time though the analysis of spectroscopic image data associated with the ejecta of impact craters [@grier2001;@taiudovicic2021]. Analysis of crater data is becoming more computationally expensive as planetary image datasets can exceed TBs and sometimes PBs in size, while crater databases can exceed 10$^6$ craters [@robbins2018;@wang2021] and perhaps orders of magnitude more with the rapid improvement of machine learning crater detection algorithms in recent years [@silburt2019;@delatte2019;@lagrassa2023]. Therefore, crater data analysts require an efficient set of tools to extract and store data or summary statistics associated with craters and their surrounding ejecta patterns. `Craterpy` was written to facilitate impact crater data analysis by producing and managing regions of interest associated with impact craters, extracting data and summary statistics from associated spectral image data, and proving a clean and modular API for users to quickly generate and analyze spatial crater data. 

# Statement of need

Most crater data analysis tasks require the conversion of lists of crater locations to precise regions of interest (ROIs) georeferenced to the planetary body of interest. This process can be time consuming and error-prone for non-specialists, particularly on large crater databases. Furthermore, the python geospatial stack is primarily tested on Earth-based geometries which can lead to errors when processing planetary image and shapefile data. `Craterpy` provides a user-friendly interface for producing and managing planetary shapefiles related to impact crater ROIs and also facilitates extraction of planetary image data associated with impact craters, and zonal statistics, for ease of use in data analytical tasks. `Craterpy` can be used to produce and plot crater regions of interest as well as the statistics associated with each ROI. 

`Craterpy` is built to be modular and interoperable with tools that planetary scientist users may already be familiar with. For instance, Craterpy is built upon the popular `geopandas` and library, and crater information can be exported to GIS shapefile formats, such as GeoJSON, for use in mapping software like QGIS and ArcGIS. The toolkit is tested on multiple planetary bodies to ensure spatially accurate data extraction and statistics calculations on planetary raster image data. `Craterpy` was designed to be used by planetary scientists and has been used in multiple scientific publications to produce crater ROI statistics underpinning their analyses [@taiudovicic2021;@chertok2023].

# Acknowledgements

This work was supported by NASA Discovery Data Analysis grant (PI: Costello).

We would like to thank our users and community for their invaluable feedback and contributions over the years.

# References