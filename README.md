# Farmland-Biodiversity-Indicator
Developing a Farmland Biodiversity Indicator using DLR CropTpe Maps , Copernicus LMS data for Bayern District Germany 


Habitats defined 
Cropland, Perennial, Grassland and Small Woody Features (including Hedgerows)

**Data to be downloaded **
    Croptype maps, Grassland Mowing Frequency , Grassland First Cut and Hedgerows : https://geoservice.dlr.de/ (2017-2024)
    Small Woody Features:  https://land.copernicus.eu/ ( 2015,2018,2021)
    Hexagon shapefiles https://github.com/uber/h3


    
 Preliminary steps needed to be done 
 1. Downloading the necessary data for the years requried for the case
 2. Select the geographical subset thats required


The codes uploaded are Chronolgically named 
1= Pre-processing (Clipping, reprojecting and resamplign ) 
2=Genreating Intermidiates( Crop Count, S/L rotation, Legumes)
3=Assigning Habitat Qiuality Scores /resampling ( editable )
4=Final 

if the names have alphanumeric numbers such as 1a, 2a they denote a sub process of the main task 

General Methodology Followed 

