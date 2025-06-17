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
  affiliation: "2"
- name: Emily S. Costello
  orcid: 0000-0001-7939-9867
  affiliation: "1"
affiliations:
- name: Hawaiʻi Institute of Geophysics and Planetology, School of Ocean and Earth Science and Technology, University of Hawaiʻi at Manoa, Honolulu, HI, United States of America
  index: 1
- name: Los Alamos National Laboratory, Los Alamos, NM, United States of America
  index: 2
date: 16 June 2025
bibliography: paper.bib
---

# Summary

`Craterpy` is an open-source Python package that reads impact crater databases, defines spatial regions of interest, and extracts data associated with each crater into standard Python data structures. It provides a simple and consistent toolkit for parsing, cleaning, and georeferencing crater data into Geopandas `GeoDataFrames`, and allows visualization and extraction of statistics from georeferenced raster imagery of planetary image datasets in tabular format. `Craterpy` simplifies many error-prone tasks like defining crater regions of interest in the correct projection, reprojecting to the format of the desired raster, and extracting data for possibly millions of craters from GB to TB-sized imagery in a computationally and memory-efficient manner.  All of these features come from a clean object-oriented user interface, built to be interoperable with the planetary geospatial Python stack and Geographic Information System (GIS) software. 

Impact craters and their ejecta, material distributed around them during formation, are commonly used to determine the composition and active processes at or below planetary surfaces. For example, craters have been used to study the rate of chemical alteration of the Moon's surface due to space exposure through the collection of iron nanoparticles on crater ejecta [@grier2001;@taiudovicic2021]. Analysis of crater data is becoming more computationally expensive as planetary image datasets can exceed TBs and sometimes PBs in size, while crater databases can exceed 10$^6$ craters [@robbins2018;@wang2021] and perhaps orders of magnitude more with the rapid improvement of machine learning crater detection algorithms in recent years [@silburt2019;@delatte2019;@lagrassa2023]. Therefore, planetary scientists require an efficient set of tools to accurately and efficiently extract and process image data associated with craters and their surrounding ejecta. 

# Statement of need

Most crater data analysis tasks require common, repetitive tasks like the conversion of crater databases to precise regions of interest (ROIs) georeferenced to the planetary body of interest. This process can be time consuming and error-prone for non-specialists, particularly on large crater databases. In addition, planetary geospatial standards and formats are can be obscure with a steep learning curve and lack of documentation and tutorial materials relative to the burgeoning Earth-based open-source geospatial stack. 

The user-friendly tools provided by `Craterpy` allow crater data analysis to be simple and fast in Python. Extracting research image data from diverse datasets for a wide number a planetary bodies becomes tractable with a clean, consistent API, while publication quality plots are achievable in a couple simple commands. `Craterpy` is also built to be modular and interoperable with tools that planetary scientist users may already be familiar with. For instance, Craterpy is built on the popular `geopandas` library, and crater information can be easily exported to common GIS formats, such as GeoJSON, for use in mapping software like QGIS or ArcGIS. While these generalist mapping tools can accomplish many of the tasks that `Craterpy` provides, they can have steep learning curves and often struggle with converting between (non-Earth) planetary projections. `Craterpy` was designed to take crater data in formats commonly used by planetary scientists and deliver statistics that enable a wide range of inquiry into planetary geology across the Solar System. `Craterpy` has been used in two published research articles thus far [@taiudovicic2021;@chertok2023] and is presently in use by multiple NASA-funded planetary data analysis grants.

# Acknowledgements

This work was supported by NASA Discovery Data Analysis grant (80NSSC24K0065, PI: Costello).

We would like to thank our users and community for their invaluable feedback and contributions over the years.

# References