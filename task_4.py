#!/usr/bin/env python
# coding: utf-8

# In[113]:


import numpy as np
import rasterio
from rasterio.plot import reshape_as_raster, reshape_as_image
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.ops import unary_union
from rasterio.features import rasterize
from shapely.geometry import mapping


# In[114]:


#Open image and display
raster_path = 'T36UXV_20200406T083559_TCI_10m.jp2'
with rasterio.open(raster_path,"r", driver = 'JP2OpenJPEG') as src:
    raster_img = src.read()
    raster_meta = src.meta


# In[115]:


raster_img = reshape_as_image(raster_img)
plt.figure(figsize=(15,15))
plt.imshow(raster_img)


# In[ ]:





# In[116]:


#Read vector mask data
train_df = gpd.read_file('mask/Masks_T36UXV_20190427.shp')
train_df[train_df.geometry.notnull()]


# In[117]:


# Redesign vector data to match raster data
train_df = train_df.to_crs(raster_meta['crs'])


# In[118]:


# Determine a function to convert polygon coordinates to raster coordinates
def poly_from_utm(polygon,transform):
    poly_pts = []
    poly = unary_union(polygon)
    for i in np.array(poly.exterior.coords):
        poly_pts.append(~transform * tuple(i))
    new_poly = Polygon(poly_pts)
    return new_poly


# In[120]:


# Convert vector polygons to raster mask
poly_shp = []
im_size = (src.meta['height'], src.meta['width'])
for num, row in train_df.iterrows():
    if row['geometry'] is None:
        continue
    if row['geometry'].geom_type == 'Polygon':
        poly = poly_from_utm(row['geometry'], src.meta['transform'])
        poly_shp.append(poly)
    else:
        for p in row['geometry']:
            poly = poly_from_utm(p, src.meta['transform'])
            poly_shp.append(poly)
mask = rasterize(shapes=poly_shp, out_shape=im_size)
#plotting the mask
plt.figure(figsize=(15,15))
plt.imshow(mask)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




